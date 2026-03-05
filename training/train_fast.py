"""
train_fast.py — CleanRL-style PPO with vectorized environments.

Key speedups over train.py:
    1. Runs N_ENVS environments in parallel (sync vectorization)
    2. Precomputes the full score lookup table at startup
    3. Larger rollout batches = better CPU/GPU utilization
    4. Exponential reward shaping

Usage:
    uv run training/train_fast.py

Expected speedup: 4-8x over train.py depending on N_ENVS.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env.wordle_env import WordleEnv


# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_DIR          = "data"
MODEL_DIR         = "models"

N_ENVS            = 32        # parallel environments
N_ITERATIONS      = 5_000     # total training iterations
STEPS_PER_ENV     = 32        # steps collected per env per iteration
                               # total batch = N_ENVS * STEPS_PER_ENV = 1024

LR                = 1e-4
HIDDEN_DIM        = 256
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_EPS          = 0.2
VF_COEF           = 0.5
ENT_COEF          = 0.005     # low — encourage commitment
MAX_GRAD_NORM     = 0.5
N_EPOCHS          = 4
MINIBATCH_SIZE    = 256

LOG_EVERY         = 25
SAVE_EVERY        = 500
EVAL_EPISODES     = 500

# Exponential win rewards — each step is worth double the next
WIN_REWARDS = {1: 32.0, 2: 16.0, 3: 8.0, 4: 4.0, 5: 2.0, 6: 1.0}
LOSS_REWARD = -6.0
STEP_PENALTY = 0.0
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Score cache — precompute all (guess, secret) → colors
# ══════════════════════════════════════════════════════════════════════════════

def build_score_cache(env: WordleEnv) -> dict:
    """
    Precompute scores[(guess_idx, secret_idx)] = tuple of colors.
    One-time cost at startup, eliminates repeated _score() calls during training.
    ~2-4 mins for 14k words. Saves to disk so subsequent runs are instant.
    """
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")

    if os.path.exists(cache_path):
        print("  Loading score cache from disk...", end=" ", flush=True)
        cache = np.load(cache_path)
        print(f"OK  ({cache.shape})")
        return cache

    V = len(env.guesses)
    print(f"  Precomputing {V}×{V} = {V*V:,} scores (one-time)...")
    print("  This takes 2-4 minutes and is saved to disk for future runs.")
    t0 = time.time()

    # cache[guess_idx, secret_idx] = encoded colors (base-3 int, 0-242)
    # Encoding: sum(color[i] * 3^i) — compact single int per pair
    cache = np.zeros((V, V), dtype=np.uint8)

    for g_idx, guess in enumerate(env.guesses):
        if g_idx % 1000 == 0:
            pct = g_idx / V * 100
            print(f"    {pct:.0f}%  ({g_idx}/{V})  {time.time()-t0:.0f}s elapsed")
        for s_idx, secret in enumerate(env.guesses):
            colors = WordleEnv._score(guess, secret)
            # Encode 5 colors (each 0-2) as single uint8 using base-3
            encoded = sum(c * (3 ** i) for i, c in enumerate(colors))
            cache[g_idx, s_idx] = encoded

    np.save(cache_path, cache)
    print(f"  Done in {(time.time()-t0)/60:.1f} min — saved to {cache_path}")
    return cache


def decode_score(encoded: int) -> list:
    """Decode base-3 encoded score back to list of 5 colors."""
    colors = []
    for _ in range(5):
        colors.append(encoded % 3)
        encoded //= 3
    return colors


# ══════════════════════════════════════════════════════════════════════════════
#  Fast Wordle Env (uses precomputed score cache)
# ══════════════════════════════════════════════════════════════════════════════

class FastWordleEnv:
    """
    Stripped-down env that uses the precomputed score cache for speed.
    Same interface as WordleEnv: reset() → (obs, mask), step() → (obs, r, done, info)
    """

    WORD_LEN    = 5
    MAX_GUESSES = 6
    EMPTY_LETTER = 26
    EMPTY_COLOR  = 3
    GREEN        = 2

    def __init__(self, base_env: WordleEnv, score_cache: np.ndarray):
        self.words       = base_env.guesses   # single unified word list
        self.vocab_size  = len(self.words)
        self.score_cache = score_cache
        self.obs_dim     = self.WORD_LEN * self.MAX_GUESSES * 2 + 1

        # Internal state
        self.secret_idx     = 0
        self.step_num       = 0
        self.board_letters  = np.zeros(self.WORD_LEN * self.MAX_GUESSES, dtype=np.int32)
        self.board_colors   = np.zeros(self.WORD_LEN * self.MAX_GUESSES, dtype=np.int32)
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
        self.done           = False

    def reset(self):
        self.secret_idx    = np.random.randint(self.vocab_size)
        self.step_num      = 0
        self.board_letters = np.full(
            self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_LETTER, dtype=np.int32
        )
        self.board_colors  = np.full(
            self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_COLOR, dtype=np.int32
        )
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
        self.done = False
        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        encoded = int(self.score_cache[action, self.secret_idx])
        colors  = decode_score(encoded)

        start = self.step_num * self.WORD_LEN
        guess = self.words[action]
        for i in range(self.WORD_LEN):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]

        self.step_num += 1
        won  = all(c == self.GREEN for c in colors)
        over = won or self.step_num >= self.MAX_GUESSES
        self.done = over

        # ── Exponential reward ─────────────────────────────────────────────
        reward = STEP_PENALTY
        if over:
            reward += WIN_REWARDS[self.step_num] if won else LOSS_REWARD

        if not self.done:
            self._recompute_mask(action, colors)

        info = {"won": won, "step": self.step_num}
        return self._obs(), reward, self.done, info

    def valid_mask(self):
        return self.valid_mask_arr.copy()

    def _recompute_mask(self, guess_idx: int, colors: list):
        encoded_target = sum(c * (3 ** i) for i, c in enumerate(colors))
        # Keep only words where scoring guess against them gives same colors
        new_mask = (
            self.score_cache[guess_idx, :] == encoded_target
        ) & self.valid_mask_arr
        self.valid_mask_arr = new_mask if new_mask.any() else np.ones(self.vocab_size, dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.board_letters.astype(np.float32) / 26.0,
            self.board_colors .astype(np.float32) /  3.0,
            np.array([self.step_num / self.MAX_GUESSES], dtype=np.float32),
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorized env wrapper — runs N envs synchronously
# ══════════════════════════════════════════════════════════════════════════════

class VecWordleEnv:
    """Synchronous vectorized environment — N independent envs stepped together."""

    def __init__(self, n_envs: int, base_env: WordleEnv, score_cache: np.ndarray):
        self.n_envs = n_envs
        self.envs   = [FastWordleEnv(base_env, score_cache) for _ in range(n_envs)]

    def reset(self):
        obs_list, mask_list = zip(*[e.reset() for e in self.envs])
        return np.stack(obs_list), np.stack(mask_list)

    def step(self, actions: np.ndarray):
        results = [self.envs[i].step(int(actions[i])) for i in range(self.n_envs)]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
            infos,
        )

    def get_masks(self):
        return np.stack([e.valid_mask() for e in self.envs])

    def reset_done(self, dones: np.ndarray):
        """Auto-reset envs that finished, return new obs+masks for those envs."""
        new_obs   = np.zeros((self.n_envs, self.envs[0].obs_dim), dtype=np.float32)
        new_masks = np.zeros((self.n_envs, self.envs[0].vocab_size), dtype=bool)
        for i, done in enumerate(dones):
            if done:
                o, m = self.envs[i].reset()
                new_obs[i]   = o
                new_masks[i] = m
        return new_obs, new_masks


# ══════════════════════════════════════════════════════════════════════════════
#  Network (identical to network.py — inlined for single-file CleanRL style)
# ══════════════════════════════════════════════════════════════════════════════

class WordleNet(nn.Module):

    def __init__(self, obs_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head  = nn.Linear(hidden_dim, 1)

        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask=None):
        h      = self.trunk(obs)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        values = self.value_head(h).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def get_action(self, obs_np, mask_np, deterministic=False):
        obs_t  = torch.FloatTensor(obs_np)
        mask_t = torch.BoolTensor(mask_np)
        logits, values = self(obs_t, mask_t)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return actions.numpy(), chosen_lp.numpy(), values.numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  CleanRL-style training loop
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, last_values, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Compute GAE advantages over a (T, N_ENVS) rollout.
    rewards, values, dones : (T, N_ENVS)
    last_values            : (N_ENVS,)
    """
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(T)):
        next_val          = last_values if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = (
            rewards[t]
            + gamma * next_val * next_non_terminal
            - values[t]
        )
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def evaluate(base_env: WordleEnv, score_cache: np.ndarray, net: WordleNet, n=500):
    wins, total_guesses = 0, 0
    env = FastWordleEnv(base_env, score_cache)
    for _ in range(n):
        obs, mask = env.reset()
        done = False
        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = torch.BoolTensor(mask).unsqueeze(0)
            logits, _ = net(obs_t, mask_t)
            action = int(logits.argmax(dim=-1).item())
            obs, _, done, info = env.step(action)
            mask = env.valid_mask()
        if info["won"]: wins += 1
        total_guesses += info["step"]
    return wins / n, total_guesses / n


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Wordle RL — Fast PPO Training  (CleanRL style)")
    print(f"{'='*60}")

    base_env    = WordleEnv(DATA_DIR)
    score_cache = build_score_cache(base_env)
    vec_env     = VecWordleEnv(N_ENVS, base_env, score_cache)

    obs_dim    = base_env.obs_dim
    vocab_size = base_env.vocab_size
    net        = WordleNet(obs_dim, vocab_size, HIDDEN_DIM)
    optimizer  = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    total_params = sum(p.numel() for p in net.parameters())
    batch_size   = N_ENVS * STEPS_PER_ENV

    print(f"  Network parameters : {total_params:,}")
    print(f"  Vocabulary size    : {vocab_size:,}")
    print(f"  Parallel envs      : {N_ENVS}")
    print(f"  Batch size         : {batch_size}  ({N_ENVS} envs × {STEPS_PER_ENV} steps)")
    print(f"  Reward scale       : {list(WIN_REWARDS.values())} / {LOSS_REWARD}")
    print(f"{'='*60}\n")

    # Initial reset
    obs, masks = vec_env.reset()

    win_history = []
    t_start     = time.time()

    for iteration in range(1, N_ITERATIONS + 1):

        # ── Rollout collection ────────────────────────────────────────────────
        mb_obs      = np.zeros((STEPS_PER_ENV, N_ENVS, obs_dim),    dtype=np.float32)
        mb_masks    = np.zeros((STEPS_PER_ENV, N_ENVS, vocab_size), dtype=bool)
        mb_actions  = np.zeros((STEPS_PER_ENV, N_ENVS),             dtype=np.int64)
        mb_log_probs= np.zeros((STEPS_PER_ENV, N_ENVS),             dtype=np.float32)
        mb_values   = np.zeros((STEPS_PER_ENV, N_ENVS),             dtype=np.float32)
        mb_rewards  = np.zeros((STEPS_PER_ENV, N_ENVS),             dtype=np.float32)
        mb_dones    = np.zeros((STEPS_PER_ENV, N_ENVS),             dtype=bool)

        ep_wins, ep_count = 0, 0

        for step in range(STEPS_PER_ENV):
            mb_obs  [step] = obs
            mb_masks[step] = masks

            actions, log_probs, values = net.get_action(obs, masks)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_masks = vec_env.get_masks()

            mb_actions  [step] = actions
            mb_log_probs[step] = log_probs
            mb_values   [step] = values
            mb_rewards  [step] = rewards
            mb_dones    [step] = dones

            # Auto-reset done envs
            for i, done in enumerate(dones):
                if done:
                    ep_count += 1
                    if infos[i]["won"]:
                        ep_wins += 1
                    r_obs, r_mask = vec_env.envs[i].reset()
                    next_obs[i]   = r_obs
                    next_masks[i] = r_mask

            obs, masks = next_obs, next_masks

        # Bootstrap last values for GAE
        _, _, last_values = net.get_action(obs, masks)
        advantages, returns = compute_gae(mb_rewards, mb_values, mb_dones, last_values)

        # ── PPO update ────────────────────────────────────────────────────────
        # Flatten (T, N_ENVS) → (T*N_ENVS,)
        b_obs      = torch.FloatTensor(mb_obs.reshape(-1, obs_dim))
        b_masks    = torch.BoolTensor (mb_masks.reshape(-1, vocab_size))
        b_actions  = torch.LongTensor (mb_actions.reshape(-1))
        b_old_lp   = torch.FloatTensor(mb_log_probs.reshape(-1))
        b_adv      = torch.FloatTensor(advantages.reshape(-1))
        b_ret      = torch.FloatTensor(returns.reshape(-1))

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_pg = total_vf = total_ent = 0.0
        n_updates = 0
        n = len(b_obs)

        for _ in range(N_EPOCHS):
            perm = torch.randperm(n)
            for start in range(0, n, MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]

                logits, values = net(b_obs[idx], b_masks[idx])
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(b_actions[idx])
                entropy  = dist.entropy()

                ratio = torch.exp(new_lp - b_old_lp[idx])
                a     = b_adv[idx]
                pg_loss = -torch.min(
                    ratio * a,
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a
                ).mean()

                vf_loss  = F.mse_loss(values, b_ret[idx])
                ent_loss = -entropy.mean()
                loss     = pg_loss + VF_COEF * vf_loss + ENT_COEF * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_pg  += pg_loss.item()
                total_vf  += vf_loss.item()
                total_ent += (-ent_loss).item()
                n_updates += 1

        # ── Logging ───────────────────────────────────────────────────────────
        win_rate = ep_wins / max(ep_count, 1)
        win_history.append(win_rate)

        if iteration % LOG_EVERY == 0:
            recent_win = np.mean(win_history[-LOG_EVERY:])
            elapsed    = time.time() - t_start
            d          = max(n_updates, 1)
            print(
                f"  iter {iteration:5d}/{N_ITERATIONS} | "
                f"win {recent_win:5.1%} | "
                f"eps {ep_count:4d} | "
                f"pg {total_pg/d:+.4f} | "
                f"vf {total_vf/d:.4f} | "
                f"ent {total_ent/d:.3f} | "
                f"{elapsed/60:.1f}m"
            )

        if iteration % (LOG_EVERY * 10) == 0:
            wr, ag = evaluate(base_env, score_cache, net, EVAL_EPISODES)
            print(f"  {'─'*54}")
            print(f"  [EVAL]  greedy win rate = {wr:.1%}  avg guesses = {ag:.2f}")
            print(f"  {'─'*54}")

        if iteration % SAVE_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"fast_checkpoint_{iteration:05d}.pt")
            torch.save({"iteration": iteration, "state_dict": net.state_dict()}, path)
            print(f"  → saved {path}")

    # ── Final ─────────────────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "fast_final.pt")
    torch.save({"state_dict": net.state_dict()}, final_path)
    wr, ag = evaluate(base_env, score_cache, net, EVAL_EPISODES)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Final greedy win rate : {wr:.1%}")
    print(f"  Final avg guesses     : {ag:.2f}")
    print(f"  Model saved           : {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()