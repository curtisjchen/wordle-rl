"""
train_fast.py — CleanRL-style PPO with vectorized environments + embedding network.

Key improvements over train.py:
    1. Embedding network — learned letter (32-dim) + color (8-dim) + step (8-dim)
       embeddings replace raw normalized floats. Network sees relationships
       between letters rather than arbitrary numbers.
    2. Information gain reward — each guess rewarded for how much it
       shrinks the candidate search space: 0.5 * log(before/after).
    3. Score cache — precomputes all scores once, saves to disk.
       _recompute_mask becomes a single numpy op instead of a Python loop.
    4. Vectorized envs — 32 environments stepped in parallel.

Usage:
    uv run training/train_fast.py

First run: ~2-4 min to build score cache (saved to data/score_cache.npy).
Subsequent runs: cache loads instantly.
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
DATA_DIR       = "data"
MODEL_DIR      = "models"

N_ENVS         = 32
N_ITERATIONS   = 5_000
STEPS_PER_ENV  = 32        # batch = N_ENVS * STEPS_PER_ENV = 1024

LR             = 1e-4
HIDDEN_DIM     = 256
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENT_COEF       = 0.005
MAX_GRAD_NORM  = 0.5
N_EPOCHS       = 4
MINIBATCH_SIZE = 256

LOG_EVERY      = 25
SAVE_EVERY     = 500
EVAL_EPISODES  = 500

# Must match wordle_env.py
WIN_REWARDS    = {1: 32.0, 2: 16.0, 3: 8.0, 4: 4.0, 5: 2.0, 6: 1.0}
LOSS_REWARD    = -6.0
INFO_GAIN_COEF =  0.5
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Score cache
# ══════════════════════════════════════════════════════════════════════════════

def build_score_cache(env: WordleEnv) -> np.ndarray:
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")

    if os.path.exists(cache_path):
        print("  Loading score cache...", end=" ", flush=True)
        cache = np.load(cache_path)
        print(f"OK  {cache.shape}")
        return cache

    V = len(env.guesses)
    print(f"  Building score cache ({V}x{V} = {V*V:,} pairs) — one time only...")
    t0    = time.time()
    cache = np.zeros((V, V), dtype=np.uint8)

    for g_idx, guess in enumerate(env.guesses):
        if g_idx % 1000 == 0:
            print(f"    {g_idx/V*100:.0f}%  ({time.time()-t0:.0f}s)")
        for s_idx, secret in enumerate(env.guesses):
            colors  = WordleEnv._score(guess, secret)
            encoded = sum(c * (3 ** i) for i, c in enumerate(colors))
            cache[g_idx, s_idx] = encoded

    np.save(cache_path, cache)
    print(f"  Done in {(time.time()-t0)/60:.1f} min — saved to {cache_path}")
    return cache


def decode_colors(encoded: int) -> list:
    colors = []
    for _ in range(5):
        colors.append(encoded % 3)
        encoded //= 3
    return colors


# ══════════════════════════════════════════════════════════════════════════════
#  Fast env (uses score cache)
# ══════════════════════════════════════════════════════════════════════════════

class FastWordleEnv:
    WORD_LEN     = 5
    MAX_GUESSES  = 6
    EMPTY_LETTER = 26
    EMPTY_COLOR  = 3
    GREEN        = 2

    def __init__(self, base_env: WordleEnv, score_cache: np.ndarray):
        self.words       = base_env.words
        self.vocab_size  = len(self.words)
        self.score_cache = score_cache
        self.obs_dim     = self.WORD_LEN * self.MAX_GUESSES * 2 + 1
        self._reset_state()

    def _reset_state(self):
        self.secret_idx     = 0
        self.step_num       = 0
        self.board_letters  = np.full(self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_LETTER, dtype=np.int32)
        self.board_colors   = np.full(self.WORD_LEN * self.MAX_GUESSES, self.EMPTY_COLOR,  dtype=np.int32)
        self.valid_mask_arr = np.ones(self.vocab_size, dtype=bool)
        self.done           = False

    def reset(self):
        self._reset_state()
        self.secret_idx = np.random.randint(self.vocab_size)
        return self._obs(), self.valid_mask_arr.copy()

    def step(self, action: int):
        encoded = int(self.score_cache[action, self.secret_idx])
        colors  = decode_colors(encoded)
        guess   = self.words[action]

        start = self.step_num * self.WORD_LEN
        for i in range(self.WORD_LEN):
            self.board_letters[start + i] = ord(guess[i]) - ord('a')
            self.board_colors [start + i] = colors[i]

        self.step_num += 1
        won  = all(c == self.GREEN for c in colors)
        over = won or self.step_num >= self.MAX_GUESSES
        self.done = over

        before = int(self.valid_mask_arr.sum())
        if not self.done:
            self._recompute_mask(action, colors)
        after = int(self.valid_mask_arr.sum())

        info_gain = np.log(before + 1) - np.log(after + 1)
        reward    = INFO_GAIN_COEF * info_gain
        if over:
            reward += WIN_REWARDS[self.step_num] if won else LOSS_REWARD

        info = {"won": won, "step": self.step_num, "candidates": after}
        return self._obs(), reward, self.done, info

    def valid_mask(self):
        return self.valid_mask_arr.copy()

    def _recompute_mask(self, guess_idx: int, colors: list):
        target           = sum(c * (3 ** i) for i, c in enumerate(colors))
        new_mask         = (self.score_cache[guess_idx, :] == target) & self.valid_mask_arr
        self.valid_mask_arr = new_mask if new_mask.any() else np.ones(self.vocab_size, dtype=bool)

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            self.board_letters,
            self.board_colors,
            np.array([self.step_num], dtype=np.int32),
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorized env
# ══════════════════════════════════════════════════════════════════════════════

class VecWordleEnv:
    def __init__(self, n_envs, base_env, score_cache):
        self.n_envs = n_envs
        self.envs   = [FastWordleEnv(base_env, score_cache) for _ in range(n_envs)]

    def reset(self):
        results = [e.reset() for e in self.envs]
        obs, masks = zip(*results)
        return np.stack(obs), np.stack(masks)

    def step(self, actions):
        results = [self.envs[i].step(int(actions[i])) for i in range(self.n_envs)]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, np.float32), np.array(dones, bool), infos

    def get_masks(self):
        return np.stack([e.valid_mask() for e in self.envs])


# ══════════════════════════════════════════════════════════════════════════════
#  Embedding network
# ══════════════════════════════════════════════════════════════════════════════

class WordleNet(nn.Module):
    """
    Embedding-based actor-critic.

    Each of the 30 board tiles is embedded as:
        letter embedding  (27 values → 32 dims)   27 = a-z + empty sentinel
        color  embedding  ( 4 values →  8 dims)    4 = gray/yellow/green/empty

    The current step is embedded separately (7 values → 8 dims).

    Final input to trunk: 30 × (32+8) + 8 = 1208 dims.

    vs. old MLP input: 61 raw floats.

    The network can now learn that vowels cluster together, that green > yellow,
    and that late steps require more targeted guesses.
    """

    LETTER_DIM = 32
    COLOR_DIM  =  8
    STEP_DIM   =  8

    def __init__(self, vocab_size: int, word_len: int = 5,
                 max_guesses: int = 6, hidden_dim: int = 256):
        super().__init__()

        self.word_len    = word_len
        self.max_guesses = max_guesses
        self.n_tiles     = word_len * max_guesses  # 30

        self.letter_embed = nn.Embedding(27, self.LETTER_DIM)
        self.color_embed  = nn.Embedding( 4, self.COLOR_DIM)
        self.step_embed   = nn.Embedding(max_guesses + 1, self.STEP_DIM)

        trunk_in = self.n_tiles * (self.LETTER_DIM + self.COLOR_DIM) + self.STEP_DIM

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim), nn.ReLU(),
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

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, 61)  int64
            [:, :30]   = letter indices  (0-26)
            [:, 30:60] = color  indices  (0-3)
            [:, 60]    = step            (0-6)
        """
        letters = obs[:, :self.n_tiles].long()
        colors  = obs[:, self.n_tiles:2 * self.n_tiles].long()
        step    = obs[:, -1].long()

        l_emb    = self.letter_embed(letters)             # (B, 30, 32)
        c_emb    = self.color_embed(colors)               # (B, 30,  8)
        s_emb    = self.step_embed(step)                  # (B, 8)

        tile_emb = torch.cat([l_emb, c_emb], dim=-1)     # (B, 30, 40)
        tile_flat = tile_emb.view(tile_emb.size(0), -1)  # (B, 1200)

        return torch.cat([tile_flat, s_emb], dim=-1)      # (B, 1208)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        h      = self.trunk(self._embed(obs))
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        values = self.value_head(h).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def get_action(self, obs_np, mask_np, deterministic=False):
        obs_t  = torch.LongTensor(obs_np)
        mask_t = torch.BoolTensor(mask_np)
        logits, values = self(obs_t, mask_t)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = Categorical(logits=logits).sample()
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return actions.cpu().numpy(), chosen_lp.cpu().numpy(), values.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  GAE
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, last_values):
    T          = len(rewards)
    advantages = np.zeros_like(rewards)
    gae        = 0.0
    for t in reversed(range(T)):
        nv    = last_values if t == T - 1 else values[t + 1]
        nt    = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + GAMMA * nv * nt - values[t]
        gae   = delta + GAMMA * GAE_LAMBDA * nt * gae
        advantages[t] = gae
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(base_env, score_cache, net, n=500):
    wins = total_guesses = total_candidates = 0
    env  = FastWordleEnv(base_env, score_cache)
    for _ in range(n):
        obs, mask = env.reset()
        done = False
        while not done:
            o = torch.LongTensor(obs).unsqueeze(0)
            m = torch.BoolTensor(mask).unsqueeze(0)
            logits, _ = net(o, m)
            action    = int(logits.argmax(dim=-1).item())
            obs, _, done, info = env.step(action)
            mask = env.valid_mask()
        if info["won"]:     wins += 1
        total_guesses    += info["step"]
        total_candidates += info["candidates"]
    return wins / n, total_guesses / n, total_candidates / n


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Wordle RL — Fast PPO | Embeddings | Info-Gain Reward")
    print(f"{'='*64}")

    base_env    = WordleEnv(DATA_DIR)
    score_cache = build_score_cache(base_env)
    vec_env     = VecWordleEnv(N_ENVS, base_env, score_cache)

    net       = WordleNet(base_env.vocab_size, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    trunk_in = net.n_tiles * (net.LETTER_DIM + net.COLOR_DIM) + net.STEP_DIM
    print(f"  Network params : {sum(p.numel() for p in net.parameters()):,}")
    print(f"  Trunk input    : {trunk_in}  (30 tiles × 40 + 8 step embedding)")
    print(f"  Vocab size     : {base_env.vocab_size:,}")
    print(f"  Parallel envs  : {N_ENVS}  |  Batch : {N_ENVS * STEPS_PER_ENV}")
    print(f"  Reward         : info_gain×{INFO_GAIN_COEF} + terminal")
    print(f"  Win rewards    : {list(WIN_REWARDS.values())}  |  loss {LOSS_REWARD}")
    print(f"{'='*64}\n")

    obs, masks  = vec_env.reset()
    win_history = []
    t_start     = time.time()

    for iteration in range(1, N_ITERATIONS + 1):

        # ── Rollout ──────────────────────────────────────────────────────────
        mb_obs      = np.zeros((STEPS_PER_ENV, N_ENVS, base_env.obs_dim), dtype=np.int32)
        mb_masks    = np.zeros((STEPS_PER_ENV, N_ENVS, base_env.vocab_size), dtype=bool)
        mb_actions  = np.zeros((STEPS_PER_ENV, N_ENVS), dtype=np.int64)
        mb_log_probs= np.zeros((STEPS_PER_ENV, N_ENVS), dtype=np.float32)
        mb_values   = np.zeros((STEPS_PER_ENV, N_ENVS), dtype=np.float32)
        mb_rewards  = np.zeros((STEPS_PER_ENV, N_ENVS), dtype=np.float32)
        mb_dones    = np.zeros((STEPS_PER_ENV, N_ENVS), dtype=bool)

        ep_wins = ep_count = 0

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

            for i, done in enumerate(dones):
                if done:
                    ep_count += 1
                    if infos[i]["won"]:
                        ep_wins += 1
                    r_obs, r_mask         = vec_env.envs[i].reset()
                    next_obs[i]           = r_obs
                    next_masks[i]         = r_mask

            obs, masks = next_obs, next_masks

        _, _, last_values = net.get_action(obs, masks)
        adv, ret = compute_gae(mb_rewards, mb_values, mb_dones, last_values)

        # ── PPO update ───────────────────────────────────────────────────────
        n         = N_ENVS * STEPS_PER_ENV
        b_obs     = torch.LongTensor (mb_obs.reshape(n, base_env.obs_dim))
        b_masks   = torch.BoolTensor (mb_masks.reshape(n, base_env.vocab_size))
        b_actions = torch.LongTensor (mb_actions.reshape(n))
        b_old_lp  = torch.FloatTensor(mb_log_probs.reshape(n))
        b_adv     = torch.FloatTensor(adv.reshape(n))
        b_ret     = torch.FloatTensor(ret.reshape(n))

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        total_pg = total_vf = total_ent = n_updates = 0

        for _ in range(N_EPOCHS):
            perm = torch.randperm(n)
            for start in range(0, n, MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]

                logits, values = net(b_obs[idx], b_masks[idx])
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(b_actions[idx])
                entropy = dist.entropy()

                ratio   = torch.exp(new_lp - b_old_lp[idx])
                a       = b_adv[idx]
                pg_loss = -torch.min(
                    ratio * a,
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a,
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
                total_ent += -ent_loss.item()
                n_updates += 1

        # ── Logging ──────────────────────────────────────────────────────────
        win_history.append(ep_wins / max(ep_count, 1))

        if iteration % LOG_EVERY == 0:
            d = max(n_updates, 1)
            print(
                f"  iter {iteration:5d}/{N_ITERATIONS} | "
                f"win {np.mean(win_history[-LOG_EVERY:]):5.1%} | "
                f"eps {ep_count:4d} | "
                f"pg {total_pg/d:+.4f} | "
                f"vf {total_vf/d:.4f} | "
                f"ent {total_ent/d:.3f} | "
                f"{(time.time()-t_start)/60:.1f}m"
            )

        if iteration % (LOG_EVERY * 10) == 0:
            wr, ag, ac = evaluate(base_env, score_cache, net, EVAL_EPISODES)
            print(f"  {'─'*60}")
            print(f"  [EVAL] win {wr:.1%}  avg guesses {ag:.2f}  candidates left {ac:.1f}")
            print(f"  {'─'*60}")

        if iteration % SAVE_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"fast_ckpt_{iteration:05d}.pt")
            torch.save({"iteration": iteration, "state_dict": net.state_dict()}, path)
            print(f"  -> saved {path}")

    # ── Final ─────────────────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "fast_final.pt")
    torch.save({"state_dict": net.state_dict()}, final_path)
    wr, ag, ac = evaluate(base_env, score_cache, net, EVAL_EPISODES)
    print(f"\n{'='*64}")
    print(f"  Training complete!")
    print(f"  Win rate              : {wr:.1%}")
    print(f"  Avg guesses           : {ag:.2f}")
    print(f"  Avg candidates left   : {ac:.1f}")
    print(f"  Saved                 : {final_path}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()