import sys
import os
import argparse
import re

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# ------------------

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import mlflow

from env.wordle_env import WordleEnv, WIN_REWARDS, LOSS_REWARD, INFO_GAIN_COEF, STEP_PENALTY
from agent.network import WordleNetwork

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_DIR       = "data"
MODEL_DIR      = "models"

# Configuration
N_ENVS         = 64            
STEPS_PER_ENV  = 32           
MINIBATCH_SIZE = 64            
N_ITERATIONS   = 10_000

# Training
LR             = 1e-5
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF       = 0.05
MAX_GRAD_NORM  = 0.5
N_EPOCHS       = 4

# Network
EMBED_DIM      = 128
N_HEADS        = 4
N_LAYERS       = 4

LOG_EVERY      = 1             
SAVE_EVERY     = 1000
# ──────────────────────────────────────────────────────────────────────────────

def load_score_cache(env: WordleEnv):
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Run train_fast.py once to build 'score_cache.npy' first!")
    print(" Loading score cache...", end=" ", flush=True)
    cache_np = np.load(cache_path)
    print(f"OK {cache_np.shape}")
    return cache_np

# ══════════════════════════════════════════════════════════════════════════════
#  NUMPY VECTORIZED ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class NumpyWordleEnv:
    def __init__(self, base_env, score_cache_np, n_envs, reward_config=None):
        self.n_envs = n_envs
        self.vocab_size = base_env.vocab_size
        
        # --- Configurable Rewards ---
        # Default to standard if nothing passed
        if reward_config is None:
            self.win_rewards = {1: 5.0, 2: 10.0, 3: 20.0, 4: 10.0, 5: 2.0, 6: 1.0}
            self.loss_reward = -20.0
            self.step_penalty = -0.5
        else:
            self.win_rewards  = reward_config.get("win_rewards")
            self.loss_reward  = reward_config.get("loss_reward")
            self.step_penalty = reward_config.get("step_penalty")

        # Read-only shared data
        self.score_cache = score_cache_np 
        self.words_int   = np.array([[ord(c) - ord('a') for c in w] for w in base_env.words], dtype=np.int32)
        
        # Game State
        self.secret_idxs = np.zeros(n_envs, dtype=np.int32)
        self.step_nums   = np.zeros(n_envs, dtype=np.int32)
        self.masks       = np.ones((n_envs, self.vocab_size), dtype=bool)
        
        self.obs_letters = np.full((n_envs, 30), 26, dtype=np.int32)
        self.obs_colors  = np.full((n_envs, 30), 3,  dtype=np.int32)
        
        self.reset_all()

    def reset_all(self):
        self.secret_idxs = np.random.randint(0, self.vocab_size, size=self.n_envs)
        self.step_nums.fill(0)
        self.masks.fill(True)
        self.obs_letters.fill(26)
        self.obs_colors.fill(3)
        return self._get_obs(), self.masks.copy()

    def _reset_indices(self, indices):
        if len(indices) == 0: return
        self.secret_idxs[indices] = np.random.randint(0, self.vocab_size, size=len(indices))
        self.step_nums[indices] = 0
        self.masks[indices] = True
        self.obs_letters[indices] = 26
        self.obs_colors[indices]  = 3

    def step(self, actions):
        scores_encoded = self.score_cache[actions, self.secret_idxs] 

        c0 = scores_encoded % 3
        c1 = (scores_encoded // 3) % 3
        c2 = (scores_encoded // 9) % 3
        c3 = (scores_encoded // 27) % 3
        c4 = (scores_encoded // 81) % 3
        current_colors = np.stack([c0, c1, c2, c3, c4], axis=1) 

        batch_ids  = np.arange(self.n_envs)
        start_cols = self.step_nums * 5
        guess_letters = self.words_int[actions] 
        
        for i in range(5):
            cols = start_cols + i
            self.obs_letters[batch_ids, cols] = guess_letters[:, i]
            self.obs_colors [batch_ids, cols] = current_colors[:, i]

        possible_secrets = (self.score_cache[actions, :] == scores_encoded[:, None])
        self.masks &= possible_secrets
        
        empty_rows = (self.masks.sum(axis=1) == 0)
        if np.any(empty_rows):
            self.masks[empty_rows] = True
            
        self.step_nums += 1
        won = (scores_encoded == 242) 
        done = won | (self.step_nums >= 6)

        # --- Dynamic Rewards ---
        rewards = np.full(self.n_envs, self.step_penalty, dtype=np.float32)
        
        # Add Win Bonuses
        # We access self.win_rewards instead of global WIN_REWARDS
        if np.any(won & done):
            rewards[won & done] += [self.win_rewards[s] for s in self.step_nums[won & done]]
        
        # Add Loss Penalty
        rewards[(~won) & done] += self.loss_reward

        # --- Metrics ---
        done_indices = np.where(done)[0]
        won_indices  = np.where(won & done)[0]
        lost_indices = np.where((~won) & done)[0]
        
        total_guesses_batch = 0.0
        if len(won_indices) > 0:
            total_guesses_batch += np.sum(self.step_nums[won_indices])
        if len(lost_indices) > 0:
            # Penalize loss as 7.0 guesses
            total_guesses_batch += len(lost_indices) * 7.0
            
        if len(done_indices) > 0:
            avg_guesses = total_guesses_batch / len(done_indices)
        else:
            avg_guesses = 0.0

        info = {
            "wins": np.sum(won & done),
            "dones": np.sum(done),
            "avg_guesses": avg_guesses
        }
        
        if len(done_indices) > 0:
            self._reset_indices(done_indices)

        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        return np.concatenate([
            self.obs_letters,
            self.obs_colors,
            self.step_nums[:, None]
        ], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # --- 1. Parse Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .pt file to resume from")
    parser.add_argument("--name", type=str, default="cpu_wordle", help="Name prefix for saved checkpoints (e.g. 'experiment_5')")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Force CPU
    device = torch.device("cpu")
    print(f"Using device: {device} (NumPy Accelerated)")

    mlflow.set_experiment("Wordle_RL_CPU")
    
    # --- 2. Configuration ---
    REWARD_CONFIG = {
        "win_rewards": {1: 1.0, 2: 3.0, 3: 3.5, 4: 1.0, 5: -0.0, 6: -1.0},
        "loss_reward": -5.0,
        "step_penalty": -0.1
    }

    base_env = WordleEnv(DATA_DIR)
    score_cache_np = load_score_cache(base_env)
    vec_env = NumpyWordleEnv(base_env, score_cache_np, N_ENVS, reward_config=REWARD_CONFIG)
    
    net = WordleNetwork(base_env.obs_dim, base_env.vocab_size, 
                        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    
    # --- 3. Resume Logic ---
    start_iter = 1
    if args.resume:
        print(f"--> Resuming from checkpoint: {args.resume}")
        if os.path.exists(args.resume):
            state_dict = torch.load(args.resume, map_location=device)
            net.load_state_dict(state_dict)
            match = re.search(r"(\d+)", args.resume)
            if match:
                start_iter = int(match.group(1)) + 1
        else:
            print(f"--> Error: Checkpoint {args.resume} not found!")
            return

    # --- 4. Run Description (The Name in MLflow) ---
    # Example: "exp5_L4_D128_H4_Ent0.05_LR0.0003"
    run_desc = f"{args.name}_L{N_LAYERS}_D{EMBED_DIM}_H{N_HEADS}_Ent{ENT_COEF}_LR{LR}"
    
    if args.resume:
        run_desc += f"_Res{start_iter}"

    with mlflow.start_run(run_name=run_desc):
        # Log all hyperparams for sorting/filtering
        mlflow.log_params({
            "name": args.name,
            "N_ENVS": N_ENVS,
            "BATCH": MINIBATCH_SIZE,
            "LR": LR,
            "LAYERS": N_LAYERS,
            "DIM": EMBED_DIM,
            "HEADS": N_HEADS,
            "ENTROPY": ENT_COEF,
            "rewards": str(REWARD_CONFIG)
        })
        
        obs, masks = vec_env.reset_all()
        t0 = time.time()
        win_history = []
        
        for iteration in range(start_iter, N_ITERATIONS + 1):
            
            # --- [Buffer Initialization] ---
            mb_obs, mb_masks, mb_actions, mb_log_probs, mb_values, mb_rewards, mb_dones = [],[],[],[],[],[],[]
            total_wins = 0
            total_eps  = 0
            total_guesses_sum = 0

            # --- [Collection Phase] ---
            for _ in range(STEPS_PER_ENV):
                with torch.no_grad():
                    o_t = torch.as_tensor(obs, dtype=torch.long)
                    m_t = torch.as_tensor(masks, dtype=torch.bool)
                    actions, log_probs, values = net.get_action(o_t, m_t)

                next_obs, rewards, dones, info = vec_env.step(actions)
                
                mb_obs.append(obs)
                mb_masks.append(masks)
                mb_actions.append(actions)
                mb_log_probs.append(log_probs)
                mb_values.append(values)
                mb_rewards.append(rewards)
                mb_dones.append(dones)
                
                total_wins += info['wins']
                total_eps  += info['dones']
                if info['dones'] > 0:
                    total_guesses_sum += info['avg_guesses'] * info['dones']
                
                obs = next_obs
                masks = vec_env.masks.copy()

            # --- [PPO Update Phase] ---
            t_obs       = torch.tensor(np.stack(mb_obs), dtype=torch.long)
            t_masks     = torch.tensor(np.stack(mb_masks), dtype=torch.bool)
            t_actions   = torch.tensor(np.stack(mb_actions), dtype=torch.long)
            t_log_probs = torch.tensor(np.stack(mb_log_probs), dtype=torch.float)
            t_values    = torch.tensor(np.stack(mb_values), dtype=torch.float)
            t_rewards   = torch.tensor(np.stack(mb_rewards), dtype=torch.float)
            t_dones     = torch.tensor(np.stack(mb_dones), dtype=torch.bool)

            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.long)
                m_t = torch.as_tensor(masks, dtype=torch.bool)
                _, _, last_values = net.get_action(o_t, m_t)
                last_values = torch.as_tensor(last_values, dtype=torch.float)

            advantages = torch.zeros_like(t_rewards)
            gae = 0.0
            for t in reversed(range(STEPS_PER_ENV)):
                next_val = last_values if t == STEPS_PER_ENV - 1 else t_values[t + 1]
                not_done = (~t_dones[t]).float()
                delta = t_rewards[t] + GAMMA * next_val * not_done - t_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * not_done * gae
                advantages[t] = gae
                
            returns = advantages + t_values
            f_obs       = t_obs.view(-1, base_env.obs_dim)
            f_masks     = t_masks.view(-1, base_env.vocab_size)
            f_actions   = t_actions.view(-1)
            f_log_probs = t_log_probs.view(-1)
            f_returns   = returns.view(-1)
            f_adv       = advantages.view(-1)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

            dataset_size = f_obs.size(0)
            indices = torch.randperm(dataset_size)
            
            epoch_pg_loss, epoch_vf_loss, epoch_entropy = [], [], []

            for _ in range(N_EPOCHS):
                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    idx = indices[start:start + MINIBATCH_SIZE]
                    logits, values = net(f_obs[idx], f_masks[idx])
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(f_actions[idx])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_lp - f_log_probs[idx])
                    surr1 = ratio * f_adv[idx]
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * f_adv[idx]
                    pg_loss = -torch.min(surr1, surr2).mean()
                    vf_loss = F.mse_loss(values, f_returns[idx])
                    loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    epoch_pg_loss.append(pg_loss.item())
                    epoch_vf_loss.append(vf_loss.item())
                    epoch_entropy.append(entropy.item())

            # --- [Logging] ---
            avg_win_rate = total_wins / total_eps if total_eps > 0 else 0.0
            avg_guesses  = total_guesses_sum / total_eps if total_eps > 0 else 0.0
            if total_eps > 0: win_history.append(avg_win_rate)

            metrics = {
                "win_rate": avg_win_rate,
                "avg_guesses": avg_guesses,
                "eps_per_iter": total_eps,
                "pg_loss": np.mean(epoch_pg_loss),
                "vf_loss": np.mean(epoch_vf_loss),
                "entropy": np.mean(epoch_entropy)
            }
            mlflow.log_metrics(metrics, step=iteration)

            if iteration % LOG_EVERY == 0:
                smooth_win = np.mean(win_history[-10:]) if win_history else 0.0
                print(f"Iter {iteration:5d} | Win: {smooth_win:.2%} | Guess: {avg_guesses:.2f} | Ent: {metrics['entropy']:.3f}")

            if iteration % SAVE_EVERY == 0:
                # Saves as: models/experiment_1_1000.pt
                path = f"{MODEL_DIR}/{args.name}_{iteration}.pt"
                torch.save(net.state_dict(), path)
                mlflow.log_artifact(path)

    # Final Save
    final_path = f"{MODEL_DIR}/{args.name}_final.pt"
    torch.save(net.state_dict(), final_path)
    mlflow.log_artifact(final_path)

if __name__ == "__main__":
    main()