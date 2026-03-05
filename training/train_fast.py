import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
import os

# Get the directory where this script is located (training/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (wordle-rl/)
project_root = os.path.dirname(script_dir)
# Add the project root to the system path so Python can find 'env' and 'agent'
sys.path.append(project_root)

from env.wordle_env import WordleEnv, WIN_REWARDS, LOSS_REWARD, INFO_GAIN_COEF
from agent.network import WordleNetwork

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_DIR       = "data"
MODEL_DIR      = "models"

# Parallelism
N_ENVS         = 128          # High number of envs for CPU throughput
STEPS_PER_ENV  = 16           # Short rollouts
MINIBATCH_SIZE = 1024         
N_ITERATIONS   = 10_000

# Training
LR             = 3e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF       = 0.01         # Higher entropy to prevent early collapse
MAX_GRAD_NORM  = 0.5
N_EPOCHS       = 4

LOG_EVERY      = 25
SAVE_EVERY     = 1000
EVAL_EPISODES  = 500
# ──────────────────────────────────────────────────────────────────────────────

def build_score_cache(env: WordleEnv) -> np.ndarray:
    """Pre-computes 3^5 scores for every word pair. Saves massive CPU time."""
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")

    if os.path.exists(cache_path):
        print(" Loading score cache...", end=" ", flush=True)
        cache = np.load(cache_path)
        print(f"OK {cache.shape}")
        return cache

    V = len(env.guesses)
    print(f" Building score cache ({V}x{V})... this takes ~2 mins one time.")
    t0    = time.time()
    cache = np.zeros((V, V), dtype=np.uint8)

    for g_idx, guess in enumerate(env.guesses):
        for s_idx, secret in enumerate(env.guesses):
            colors  = WordleEnv._score(guess, secret)
            # Encode [2,0,1,0,0] -> scalar 0-242
            encoded = sum(c * (3 ** i) for i, c in enumerate(colors))
            cache[g_idx, s_idx] = encoded

    np.save(cache_path, cache)
    print(f" Done in {(time.time()-t0):.1f}s.")
    return cache

def decode_colors(encoded: int) -> list:
    colors = []
    for _ in range(5):
        colors.append(encoded % 3)
        encoded //= 3
    return colors

# ══════════════════════════════════════════════════════════════════════════════
#  Fast Env (Uses Cache)
# ══════════════════════════════════════════════════════════════════════════════

class FastWordleEnv:
    def __init__(self, base_env: WordleEnv, score_cache: np.ndarray):
        self.words       = base_env.words
        self.vocab_size  = len(self.words)
        self.score_cache = score_cache
        self.obs_dim     = base_env.obs_dim
        self._reset_state()

    def _reset_state(self):
        self.secret_idx     = 0
        self.step_num       = 0
        self.board_letters  = np.full(30, 26, dtype=np.int32) # 26 = empty
        self.board_colors   = np.full(30, 3,  dtype=np.int32) # 3 = empty
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
        self.done           = False

    def reset(self):
        self._reset_state()
        self.secret_idx = np.random.randint(self.vocab_size)
        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        # 1. Get Precomputed Result
        encoded = int(self.score_cache[action, self.secret_idx])
        colors  = decode_colors(encoded)
        guess   = self.words[action]

        # 2. Update Board
        start = self.step_num * 5
        for i in range(5):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]
        
        # 3. Calculate Reward
        won = all(c == 2 for c in colors)
        self.step_num += 1
        over = won or self.step_num >= 6
        self.done = over

        before = int(self.valid_mask_arr.sum())
        
        # Update valid mask (Filter allowed words based on clues)
        if not self.done:
            # Vectorized numpy filter: Keep words that would generate the same colors
            # if they were the secret
            possible_secrets = self.score_cache[action, :] == encoded
            self.valid_mask_arr &= possible_secrets
            
            # Safety: if mask is empty (shouldn't happen in logic), reset it
            if not self.valid_mask_arr.any():
                self.valid_mask_arr[:] = True

        after = int(self.valid_mask_arr.sum())

        # Main Reward: Information Gain
        info_gain = np.log(before + 1) - np.log(after + 1)
        reward    = INFO_GAIN_COEF * info_gain
        
        # Terminal Reward
        if over:
            reward += WIN_REWARDS.get(self.step_num, 0) if won else LOSS_REWARD

        info = {"won": won, "step": self.step_num, "candidates": after}
        return self._obs(), reward, self.done, info

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.board_letters,
            self.board_colors,
            np.array([self.step_num], dtype=np.int32),
        ])

class VecWordleEnv:
    def __init__(self, n_envs, base_env, score_cache):
        self.envs = [FastWordleEnv(base_env, score_cache) for _ in range(n_envs)]

    def reset(self):
        obs_list, mask_list = zip(*[e.reset() for e in self.envs])
        return np.stack(obs_list), np.stack(mask_list)

    def step(self, actions):
        results = [self.envs[i].step(int(actions[i])) for i in range(len(self.envs))]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, np.float32), np.array(dones, bool), infos

# ══════════════════════════════════════════════════════════════════════════════
#  PPO Utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, last_values):
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        nv = last_values if t == T - 1 else values[t + 1]
        nt = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + GAMMA * nv * nt - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * nt * gae
        advantages[t] = gae
    return advantages, advantages + values

def evaluate(base_env, score_cache, net, n=100):
    wins = 0
    total_guesses = 0
    env = FastWordleEnv(base_env, score_cache)
    
    for _ in range(n):
        obs, mask = env.reset()
        done = False
        while not done:
            o = torch.LongTensor(obs).unsqueeze(0)
            m = torch.BoolTensor(mask).unsqueeze(0)
            actions, _, _ = net.get_action(o, m, deterministic=True)
            obs, _, done, info = env.step(int(actions[0]))
            mask = env.valid_mask_arr
        if info["won"]: wins += 1
        total_guesses += info["step"]
        
    return wins / n, total_guesses / n

# ══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_env    = WordleEnv(DATA_DIR)
    score_cache = build_score_cache(base_env)
    vec_env     = VecWordleEnv(N_ENVS, base_env, score_cache)

    net = WordleNetwork(base_env.obs_dim, base_env.vocab_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    print(f"Network Params: {sum(p.numel() for p in net.parameters()):,}")

    obs, masks = vec_env.reset()
    win_history = []
    
    t0 = time.time()

    for iteration in range(1, N_ITERATIONS + 1):
        
        # --- Collection ---
        mb_obs      = []
        mb_masks    = []
        mb_actions  = []
        mb_log_probs= []
        mb_values   = []
        mb_rewards  = []
        mb_dones    = []

        ep_wins = 0
        ep_count = 0

        for _ in range(STEPS_PER_ENV):
            with torch.no_grad():
                o_t = torch.tensor(obs, device=device)
                m_t = torch.tensor(masks, device=device)
                actions, log_probs, values = net.get_action(o_t, m_t)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_masks = np.stack([e.valid_mask_arr for e in vec_env.envs])

            # Track History
            mb_obs.append(obs)
            mb_masks.append(masks)
            mb_actions.append(actions)
            mb_log_probs.append(log_probs)
            mb_values.append(values)
            mb_rewards.append(rewards)
            mb_dones.append(dones)

            # Auto-reset
            for i, done in enumerate(dones):
                if done:
                    ep_count += 1
                    if infos[i]["won"]: ep_wins += 1
                    r_obs, r_mask = vec_env.envs[i].reset()
                    next_obs[i] = r_obs
                    next_masks[i] = r_mask

            obs, masks = next_obs, next_masks

        # --- GAE ---
        mb_obs       = np.array(mb_obs)
        mb_masks     = np.array(mb_masks)
        mb_actions   = np.array(mb_actions)
        mb_log_probs = np.array(mb_log_probs)
        mb_values    = np.array(mb_values)
        mb_rewards   = np.array(mb_rewards)
        mb_dones     = np.array(mb_dones)

        with torch.no_grad():
            o_t = torch.tensor(obs, device=device)
            m_t = torch.tensor(masks, device=device)
            _, _, last_values = net.get_action(o_t, m_t)
        
        adv, ret = compute_gae(mb_rewards, mb_values, mb_dones, last_values)

        # Flatten
        b_obs = torch.tensor(mb_obs.reshape(-1, base_env.obs_dim), device=device)
        b_masks = torch.tensor(mb_masks.reshape(-1, base_env.vocab_size), device=device)
        b_actions = torch.tensor(mb_actions.flatten(), device=device)
        b_log_probs = torch.tensor(mb_log_probs.flatten(), device=device)
        b_adv = torch.tensor(adv.flatten(), device=device)
        b_ret = torch.tensor(ret.flatten(), device=device)

        # Normalize Advantage
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # --- Update ---
        dataset_size = b_obs.size(0)
        inds = np.arange(dataset_size)
        
        for _ in range(N_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                idx = inds[start:end]

                logits, values = net(b_obs[idx], b_masks[idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_log_probs[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv[idx]
                
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(values, b_ret[idx])
                
                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- Log ---
        if ep_count > 0:
            win_history.append(ep_wins/ep_count)
        
        if iteration == 0 or iteration % LOG_EVERY == 0:
            curr_win = np.mean(win_history[-LOG_EVERY:]) if win_history else 0.0
            print(f"Iter {iteration} | Win Rate: {curr_win:.2%} | Eps: {ep_count} | Time: {(time.time()-t0)/60:.1f}m")

        if iteration % SAVE_EVERY == 0:
            torch.save(net.state_dict(), f"{MODEL_DIR}/wordle_{iteration}.pt")
            wr, ag = evaluate(base_env, score_cache, net)
            print(f"--> EVAL: Win {wr:.1%} | Avg Guesses {ag:.2f}")

if __name__ == "__main__":
    main()