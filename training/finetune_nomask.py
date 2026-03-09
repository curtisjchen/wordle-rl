import sys
import os
import argparse
import math

# --- PATH SETUP ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
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

from env.wordle_env import WordleEnv
from agent.network import WordleNetwork

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "models"

N_ENVS         = 256
STEPS_PER_ENV  = 16
MINIBATCH_SIZE = 256
N_ITERATIONS   = 5_000

LR     = 3e-5
LR_MIN = 1e-6

GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
VF_COEF       = 0.5
N_EPOCHS      = 4
MAX_GRAD_NORM = 0.5

ENT_COEF_START = 0.01
ENT_COEF_END   = 0.001

HIDDEN_DIM = 384
MASK_PROB  = 0.0   # no scaffolding — full free exploration

# ── Intermediate rewards ───────────────────────────────────────────────────────
# Per-step reward for NEW information gained this guess.
# Keeps gradient signal flowing even on losing games.
# Max possible: 5 new greens × 0.3 = 1.5 per step — small vs terminal rewards.
GREEN_REWARD  = 0.3
YELLOW_REWARD = 0.1

# ── Fixed win rewards ──────────────────────────────────────────────────────────
# Guesses 1-4 always strongly rewarded — sharp solving always good.
WIN_REWARDS_FIXED = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0}

# ── Adaptive rewards — updated at eval time based on actual performance ────────
#
# Both guess 5/6 win rewards AND loss_reward scale together so that
# intermediate green/yellow rewards are never drowned out early in training.
#
# avg_guesses   loss    guess5   guess6
#     7.0        -1.0   +1.0     +1.0   <- struggling: celebrate anything
#     6.0        -2.0   +0.75    +0.5
#     5.0        -3.0   +0.5      0.0   <- decent: 6-guess wins break even
#     4.0        -4.0   +0.25   -0.5
#     3.0        -5.0    0.0    -1.0    <- excellent: only <=4 guess wins profit
#
# At avg_guesses=7.0 a losing game still gains from greens/yellows.
# At avg_guesses=3.0 loss is harshly penalised — model must win and win fast.

LOG_EVERY  = 1
SAVE_EVERY = 500
# ──────────────────────────────────────────────────────────────────────────────


def get_adaptive_rewards(avg_guesses: float) -> tuple[dict, float]:
    """
    Returns (win_rewards dict, loss_reward float) scaled to current performance.
    avg_guesses: 7.0 = never winning, 3.0 = excellent (both inclusive).
    """
    # Map avg_guesses 7.0->3.0 onto progress 0.0->1.0
    progress = float(np.clip((7.0 - avg_guesses) / (7.0 - 3.0), 0.0, 1.0))

    win_rewards = {
        **WIN_REWARDS_FIXED,
        5: 1.0 - 1.0 * progress,        # +1.0 -> 0.0
        6: 1.0 - 2.0 * progress,        # +1.0 -> -1.0
    }
    loss_reward = -1.0 - 4.0 * progress  # -1.0 -> -5.0

    return win_rewards, loss_reward


def load_score_cache(env: WordleEnv):
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("score_cache.npy not found.")
    print(" Loading score cache...", end=" ", flush=True)
    cache_np = np.load(cache_path).astype(np.int16)
    print(f"OK {cache_np.shape}  dtype={cache_np.dtype}")
    return cache_np


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorised environment
# ══════════════════════════════════════════════════════════════════════════════

class NumpyWordleEnv:
    """
    Vectorised Wordle with:
      - Intermediate green/yellow rewards every step for new information
      - Adaptive win rewards + loss reward (both updated via set_rewards())
      - No mask — full free exploration
      - Secrets from test_indices (common words) if provided
    """

    def __init__(self, base_env, score_cache_np, n_envs, test_indices=None):
        self.n_envs       = n_envs
        self.vocab_size   = base_env.vocab_size
        self.test_indices = test_indices
        self.mask_prob    = MASK_PROB

        # Initialise rewards at worst-case performance (wide goalposts)
        self.win_rewards, self.loss_reward = get_adaptive_rewards(7.0)

        self.score_cache = score_cache_np

        self.words_int = np.array(
            [[ord(c) - ord('a') for c in w] for w in base_env.words],
            dtype=np.int32
        )

        self.secret_idxs = np.zeros(n_envs, dtype=np.int32)
        self.step_nums   = np.zeros(n_envs, dtype=np.int32)
        self.masks       = np.ones((n_envs, self.vocab_size), dtype=bool)

        self.letter_gray       = np.zeros((n_envs, 26),    dtype=np.float32)
        self.letter_present    = np.zeros((n_envs, 26),    dtype=np.float32)
        self.letter_green      = np.zeros((n_envs, 5, 26), dtype=np.float32)
        self.letter_yellow_not = np.zeros((n_envs, 5, 26), dtype=np.float32)

        self.reset_all()

    def set_rewards(self, avg_guesses: float):
        """Update win and loss rewards based on current eval performance."""
        self.win_rewards, self.loss_reward = get_adaptive_rewards(avg_guesses)

    def _sample_secrets(self, size):
        if self.test_indices is not None:
            return np.random.choice(self.test_indices, size=size)
        return np.random.randint(0, self.vocab_size, size=size)

    def reset_all(self):
        self.secret_idxs = self._sample_secrets(self.n_envs)
        self.step_nums.fill(0)
        self.masks.fill(True)
        self.letter_gray.fill(0.0)
        self.letter_present.fill(0.0)
        self.letter_green.fill(0.0)
        self.letter_yellow_not.fill(0.0)
        return self._get_obs(), self._get_action_masks()

    def _reset_indices(self, indices):
        if len(indices) == 0:
            return
        self.secret_idxs[indices] = self._sample_secrets(len(indices))
        self.step_nums[indices]   = 0
        self.masks[indices]       = True
        self.letter_gray[indices]       = 0.0
        self.letter_present[indices]    = 0.0
        self.letter_green[indices]      = 0.0
        self.letter_yellow_not[indices] = 0.0

    def step(self, actions):
        scores_encoded = self.score_cache[actions, self.secret_idxs].astype(np.int32)

        c0 = scores_encoded % 3
        c1 = (scores_encoded // 3)  % 3
        c2 = (scores_encoded // 9)  % 3
        c3 = (scores_encoded // 27) % 3
        c4 = (scores_encoded // 81) % 3
        current_colors = np.stack([c0, c1, c2, c3, c4], axis=1)

        guess_letters = self.words_int[actions]
        env_ids = np.arange(self.n_envs)

        # Snapshot knowledge before update to count NEW information only
        prev_green_count  = self.letter_green.sum(axis=(1, 2))
        prev_yellow_count = np.clip(
            self.letter_present - self.letter_green.max(axis=1), 0, None
        ).sum(axis=1)

        # Update knowledge state
        for pos in range(5):
            l_pos = guess_letters[:, pos]
            c_pos = current_colors[:, pos]

            gray_mask   = c_pos == 0
            yellow_mask = c_pos == 1
            green_mask  = c_pos == 2

            self.letter_gray[env_ids[gray_mask], l_pos[gray_mask]] = 1.0

            self.letter_present[env_ids[yellow_mask], l_pos[yellow_mask]]         = 1.0
            self.letter_yellow_not[env_ids[yellow_mask], pos, l_pos[yellow_mask]] = 1.0

            self.letter_present[env_ids[green_mask], l_pos[green_mask]]    = 1.0
            self.letter_green[env_ids[green_mask], pos, l_pos[green_mask]] = 1.0

        new_green_count  = self.letter_green.sum(axis=(1, 2))
        new_yellow_count = np.clip(
            self.letter_present - self.letter_green.max(axis=1), 0, None
        ).sum(axis=1)

        new_greens  = np.clip(new_green_count  - prev_green_count,  0, None)
        new_yellows = np.clip(new_yellow_count - prev_yellow_count, 0, None)

        # Intermediate reward: dense signal every step
        rewards = (
            GREEN_REWARD  * new_greens +
            YELLOW_REWARD * new_yellows
        ).astype(np.float32)

        # Update candidate masks
        consistent = (
            self.score_cache[actions, :].astype(np.int32)
            == scores_encoded[:, None]
        )
        self.masks &= consistent
        empty_rows = self.masks.sum(axis=1) == 0
        if np.any(empty_rows):
            self.masks[empty_rows] = True

        self.step_nums += 1
        won  = (scores_encoded == 242)
        done = won | (self.step_nums >= 6)

        win_done  = won & done
        loss_done = (~won) & done

        # Terminal rewards on top of intermediate
        if np.any(win_done):
            rewards[win_done] += np.array(
                [self.win_rewards[s] for s in self.step_nums[win_done]],
                dtype=np.float32
            )
        rewards[loss_done] += self.loss_reward

        done_indices = np.where(done)[0]
        won_indices  = np.where(win_done)[0]
        lost_indices = np.where(loss_done)[0]

        guess_counts = np.zeros(7, dtype=np.int32)
        if len(won_indices) > 0:
            for g in self.step_nums[won_indices]:
                guess_counts[g - 1] += 1
        guess_counts[6] = len(lost_indices)

        total_guesses = 0.0
        if len(won_indices)  > 0: total_guesses += self.step_nums[won_indices].sum()
        if len(lost_indices) > 0: total_guesses += len(lost_indices) * 7.0

        info = {
            "wins":         int(np.sum(win_done)),
            "dones":        int(np.sum(done)),
            "avg_guesses":  total_guesses / len(done_indices) if len(done_indices) > 0 else 0.0,
            "guess_counts": guess_counts,
        }

        if len(done_indices) > 0:
            self._reset_indices(done_indices)

        return self._get_obs(), rewards, done, info

    def _get_obs(self) -> np.ndarray:
        step_frac = (self.step_nums / 6.0).astype(np.float32)[:, None]
        return np.concatenate([
            self.letter_gray,
            self.letter_present,
            self.letter_green.reshape(self.n_envs, -1),
            self.letter_yellow_not.reshape(self.n_envs, -1),
            step_frac,
        ], axis=1)

    def _get_action_masks(self) -> np.ndarray:
        return np.ones((self.n_envs, self.vocab_size), dtype=bool)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation — deterministic, no mask
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(net, eval_env: NumpyWordleEnv, n_games: int = 512) -> tuple[float, float]:
    obs, _ = eval_env.reset_all()
    wins, dones, guesses_sum = 0, 0, 0.0

    while dones < n_games:
        masks = np.ones((eval_env.n_envs, eval_env.vocab_size), dtype=bool)
        actions, _, _ = net.get_action(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.bool),
            deterministic=True
        )
        obs, _, done, info = eval_env.step(actions)
        wins        += info["wins"]
        dones       += info["dones"]
        guesses_sum += info["avg_guesses"] * info["dones"]

    return wins / dones, guesses_sum / dones


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (omit to train from scratch)")
    parser.add_argument("--name",  type=str, default="finetune_nomask")
    parser.add_argument("--iters", type=int, default=N_ITERATIONS)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Mask:    {MASK_PROB} (no scaffolding)")
    print(f"LR:      {LR} -> {LR_MIN} (cosine decay)")
    print(f"Entropy: {ENT_COEF_START} -> {ENT_COEF_END} (linear decay)")
    print(f"Rewards: +{GREEN_REWARD}/new green  +{YELLOW_REWARD}/new yellow  (per step)")
    print(f"         adaptive win/loss — tighten as eval avg_guesses improves")
    print(f"Secrets: common words only\n")

    mlflow.set_experiment("Wordle_RL_CPU")

    base_env    = WordleEnv(DATA_DIR)
    score_cache = load_score_cache(base_env)

    vec_env  = NumpyWordleEnv(base_env, score_cache, N_ENVS,
                               test_indices=base_env.test_indices)
    eval_env = NumpyWordleEnv(base_env, score_cache, N_ENVS,
                               test_indices=base_env.test_indices)

    net = WordleNetwork(base_env.obs_dim, base_env.vocab_size, hidden_dim=HIDDEN_DIM).to(device)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — starting from scratch")

    import sys as _sys
    if _sys.platform in ("win32", "darwin"):
        print(f"torch.compile: skipped ({_sys.platform})")
    else:
        try:
            net = torch.compile(net)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, eps=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=LR_MIN
    )

    run_desc = f"{args.name}_nomask_adaptreward_LR{LR}"
    print(f"\nStarting run: {run_desc}\n")

    with mlflow.start_run(run_name=run_desc):
        mlflow.log_params({
            "name":          args.name,
            "checkpoint":    args.checkpoint or "scratch",
            "obs_dim":       base_env.obs_dim,
            "N_ENVS":        N_ENVS,
            "STEPS_PER_ENV": STEPS_PER_ENV,
            "N_ITERATIONS":  args.iters,
            "LR":            LR,
            "ENT_START":     ENT_COEF_START,
            "ENT_END":       ENT_COEF_END,
            "GREEN_REWARD":  GREEN_REWARD,
            "YELLOW_REWARD": YELLOW_REWARD,
            "eval_set_size": len(base_env.test_indices),
            "secret_source": "common_words",
        })

        obs, masks = vec_env.reset_all()
        t0 = time.time()
        win_history = []
        guess_dist  = np.zeros(7, dtype=np.int32)

        # Baseline eval — sets initial rewards based on actual checkpoint quality
        eval_win, eval_guess = evaluate(net, eval_env)
        vec_env.set_rewards(eval_guess)
        mlflow.log_metrics({"eval_win_rate": eval_win, "eval_avg_guesses": eval_guess}, step=0)
        print(f"  -> BASELINE EVAL: Win {eval_win:.2%}  Avg guesses {eval_guess:.3f}")
        print(f"     Rewards set: loss={vec_env.loss_reward:.2f}  "
              f"guess5={vec_env.win_rewards[5]:+.2f}  "
              f"guess6={vec_env.win_rewards[6]:+.2f}\n")

        for iteration in range(1, args.iters + 1):

            progress = (iteration - 1) / max(args.iters - 1, 1)
            ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * progress

            # ── Collection ────────────────────────────────────────────
            mb_obs, mb_masks, mb_actions, mb_log_probs = [], [], [], []
            mb_values, mb_rewards, mb_dones             = [], [], []

            total_wins        = 0
            total_eps         = 0
            total_guesses_sum = 0.0
            guess_dist.fill(0)

            for _ in range(STEPS_PER_ENV):
                with torch.no_grad():
                    o_t = torch.as_tensor(obs,   dtype=torch.float32)
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

                total_wins        += info["wins"]
                total_eps         += info["dones"]
                guess_dist        += info["guess_counts"]
                if info["dones"] > 0:
                    total_guesses_sum += info["avg_guesses"] * info["dones"]

                obs   = next_obs
                masks = vec_env._get_action_masks()

            # ── PPO Update ────────────────────────────────────────────
            t_obs       = torch.tensor(np.stack(mb_obs),       dtype=torch.float32)
            t_actions   = torch.tensor(np.stack(mb_actions),   dtype=torch.long)
            t_log_probs = torch.tensor(np.stack(mb_log_probs), dtype=torch.float32)
            t_values    = torch.tensor(np.stack(mb_values),    dtype=torch.float32)
            t_rewards   = torch.tensor(np.stack(mb_rewards),   dtype=torch.float32)
            t_dones     = torch.tensor(np.stack(mb_dones),     dtype=torch.bool)

            with torch.no_grad():
                _, _, last_values = net.get_action(
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(masks, dtype=torch.bool)
                )
                last_values = torch.as_tensor(last_values, dtype=torch.float32)

            # GAE
            advantages = torch.zeros_like(t_rewards)
            gae = 0.0
            for t in reversed(range(STEPS_PER_ENV)):
                next_val = last_values if t == STEPS_PER_ENV - 1 else t_values[t + 1]
                not_done = (~t_dones[t]).float()
                delta    = t_rewards[t] + GAMMA * next_val * not_done - t_values[t]
                gae      = delta + GAMMA * GAE_LAMBDA * not_done * gae
                advantages[t] = gae

            returns = advantages + t_values

            f_obs        = t_obs.view(-1, base_env.obs_dim)
            f_actions    = t_actions.view(-1)
            f_log_probs  = t_log_probs.view(-1)
            f_values_old = t_values.view(-1)
            f_returns    = returns.view(-1)
            f_adv        = advantages.view(-1)

            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

            ret_mean       = f_returns.mean()
            ret_std        = f_returns.std() + 1e-8
            f_returns_norm = (f_returns - ret_mean) / ret_std

            dataset_size = f_obs.size(0)
            epoch_pg_loss, epoch_vf_loss, epoch_entropy = [], [], []

            for _ in range(N_EPOCHS):
                indices = torch.randperm(dataset_size)
                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    idx = indices[start: start + MINIBATCH_SIZE]

                    logits, values = net(f_obs[idx], mask=None)
                    dist    = Categorical(logits=logits)
                    new_lp  = dist.log_prob(f_actions[idx])
                    entropy = dist.entropy().mean()

                    ratio   = torch.exp(new_lp - f_log_probs[idx])
                    surr1   = ratio * f_adv[idx]
                    surr2   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * f_adv[idx]
                    pg_loss = -torch.min(surr1, surr2).mean()

                    values_norm = (values - ret_mean) / ret_std
                    v_old_norm  = (f_values_old[idx] - ret_mean) / ret_std
                    v_clipped   = v_old_norm + torch.clamp(
                        values_norm - v_old_norm, -CLIP_EPS, CLIP_EPS
                    )
                    vf_loss = torch.max(
                        F.mse_loss(values_norm, f_returns_norm[idx]),
                        F.mse_loss(v_clipped,   f_returns_norm[idx])
                    )

                    loss = pg_loss + VF_COEF * vf_loss - ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                    epoch_pg_loss.append(pg_loss.item())
                    epoch_vf_loss.append(vf_loss.item())
                    epoch_entropy.append(entropy.item())

            scheduler.step()

            # ── Logging ───────────────────────────────────────────────
            avg_win_rate = total_wins / total_eps if total_eps > 0 else 0.0
            avg_guesses  = total_guesses_sum / total_eps if total_eps > 0 else 0.0
            if total_eps > 0:
                win_history.append(avg_win_rate)

            dist_total = guess_dist.sum()
            dist_norm  = guess_dist / dist_total if dist_total > 0 else np.zeros(7)
            dist_str   = "[" + "  ".join(f"{v:.2f}" for v in dist_norm) + "]"

            metrics = {
                "win_rate":     avg_win_rate,
                "avg_guesses":  avg_guesses,
                "eps_per_iter": total_eps,
                "pg_loss":      np.mean(epoch_pg_loss),
                "vf_loss":      np.mean(epoch_vf_loss),
                "entropy":      np.mean(epoch_entropy),
                "ent_coef":     ent_coef,
                "lr":           scheduler.get_last_lr()[0],
                "loss_reward":  vec_env.loss_reward,
                "win_reward_5": vec_env.win_rewards[5],
                "win_reward_6": vec_env.win_rewards[6],
            }
            for i, v in enumerate(dist_norm):
                label = f"dist_guess_{i + 1}" if i < 6 else "dist_loss"
                metrics[label] = float(v)

            mlflow.log_metrics(metrics, step=iteration)

            if iteration % LOG_EVERY == 0:
                smooth_win = np.mean(win_history[-10:]) if win_history else 0.0
                elapsed    = time.time() - t0
                print(
                    f"Iter {iteration:5d} | "
                    f"Win: {smooth_win:.2%} | "
                    f"Guess: {avg_guesses:.2f} | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"Ent: {metrics['entropy']:.3f} | "
                    f"R5/R6/L: {vec_env.win_rewards[5]:+.2f}/{vec_env.win_rewards[6]:+.2f}/{vec_env.loss_reward:.2f} | "
                    f"t: {elapsed:.0f}s"
                )
                print(f"         Dist  1     2     3     4     5     6     L")
                print(f"              {dist_str}")

            if iteration % 250 == 0:
                eval_win, eval_guess = evaluate(net, eval_env)
                # Update rewards based on new performance
                vec_env.set_rewards(eval_guess)
                mlflow.log_metrics(
                    {"eval_win_rate":  eval_win,
                     "eval_avg_guesses": eval_guess,
                     "loss_reward":    vec_env.loss_reward,
                     "win_reward_5":   vec_env.win_rewards[5],
                     "win_reward_6":   vec_env.win_rewards[6]},
                    step=iteration
                )
                print(f"  -> EVAL: Win {eval_win:.2%}  Avg guesses {eval_guess:.3f}")
                print(f"     Rewards updated: loss={vec_env.loss_reward:.2f}  "
                      f"guess5={vec_env.win_rewards[5]:+.2f}  "
                      f"guess6={vec_env.win_rewards[6]:+.2f}")

            if iteration % SAVE_EVERY == 0:
                path  = f"{MODEL_DIR}/{args.name}_{iteration}.pt"
                state = getattr(net, "_orig_mod", net).state_dict()
                torch.save(state, path)
                mlflow.log_artifact(path)

    final_path = f"{MODEL_DIR}/{args.name}_final.pt"
    state = getattr(net, "_orig_mod", net).state_dict()
    torch.save(state, final_path)
    mlflow.log_artifact(final_path)
    print(f"\nFinetuning complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()