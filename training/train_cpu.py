import sys
import os
import argparse
import re

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

# Collection
N_ENVS         = 128
STEPS_PER_ENV  = 32
MINIBATCH_SIZE = 256
N_ITERATIONS   = 20_000

# Optimiser
LR = 3e-4

# PPO
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
VF_COEF       = 0.5
N_EPOCHS      = 6
MAX_GRAD_NORM = 0.5

# Entropy: linearly anneals from START → END over all iterations
ENT_COEF_START = 0.05
ENT_COEF_END   = 0.005

# Network
HIDDEN_DIM = 768

LOG_EVERY  = 1
SAVE_EVERY = 1000
# ──────────────────────────────────────────────────────────────────────────────


def load_score_cache(env: WordleEnv):
    cache_path = os.path.join(DATA_DIR, "score_cache.npy")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Run train_fast.py once to build 'score_cache.npy' first!")
    print(" Loading score cache...", end=" ", flush=True)
    cache_np = np.load(cache_path).astype(np.int16)
    print(f"OK {cache_np.shape}  dtype={cache_np.dtype}")
    return cache_np


# ══════════════════════════════════════════════════════════════════════════════
#  NUMPY VECTORISED ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class NumpyWordleEnv:
    """
    Fully vectorised Wordle over N_ENVS parallel games.

    Observation (float32, shape [n_envs, 183]):
        [0  :26 ]  letter_gray     — letter confirmed absent
        [26 :52 ]  letter_present  — letter confirmed present (yellow or green)
        [52 :182]  letter_green    — 5x26 grid: letter confirmed green at position p
        [182:183]  step_frac       — step_num / 6
    """

    def __init__(self, base_env, score_cache_np, n_envs, reward_config=None):
        self.n_envs     = n_envs
        self.vocab_size = base_env.vocab_size

        if reward_config is None:
            self.win_rewards  = {1: 1.0, 2: 3.0, 3: 3.5, 4: 1.0, 5: 0.5, 6: -1.0}
            self.loss_reward  = -5.0
            self.step_penalty = -0.1
        else:
            self.win_rewards  = reward_config["win_rewards"]
            self.loss_reward  = reward_config["loss_reward"]
            self.step_penalty = reward_config["step_penalty"]

        self.score_cache = score_cache_np
        self.vocab_limit = self.vocab_size  # updated externally by curriculum
        self.words_int = np.array(
            [[ord(c) - ord('a') for c in w] for w in base_env.words],
            dtype=np.int32
        )

        self.secret_idxs = np.zeros(n_envs, dtype=np.int32)
        self.step_nums   = np.zeros(n_envs, dtype=np.int32)
        self.masks       = np.ones((n_envs, self.vocab_size), dtype=bool)

        self.letter_gray    = np.zeros((n_envs, 26),    dtype=np.float32)
        self.letter_present = np.zeros((n_envs, 26),    dtype=np.float32)
        self.letter_green   = np.zeros((n_envs, 5, 26), dtype=np.float32)

        self.reset_all()

    def reset_all(self):
        self.secret_idxs = np.random.randint(0, self.vocab_limit, size=self.n_envs)
        self.step_nums.fill(0)
        self.masks.fill(True)
        self.letter_gray.fill(0.0)
        self.letter_present.fill(0.0)
        self.letter_green.fill(0.0)
        return self._get_obs(), self.masks.copy()

    def _reset_indices(self, indices):
        if len(indices) == 0:
            return
        self.secret_idxs[indices] = np.random.randint(0, self.vocab_limit, size=len(indices))
        self.step_nums[indices]   = 0
        self.masks[indices]       = True
        self.letter_gray[indices]    = 0.0
        self.letter_present[indices] = 0.0
        self.letter_green[indices]   = 0.0

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
        for pos in range(5):
            l_pos = guess_letters[:, pos]
            c_pos = current_colors[:, pos]

            gray_mask = c_pos == 0
            self.letter_gray[env_ids[gray_mask], l_pos[gray_mask]] = 1.0

            present_mask = c_pos >= 1
            self.letter_present[env_ids[present_mask], l_pos[present_mask]] = 1.0

            green_mask = c_pos == 2
            self.letter_green[env_ids[green_mask], pos, l_pos[green_mask]] = 1.0

        # Masks not updated during training — saves ~70% of step time.
        # Agent learns word elimination implicitly from rewards.

        self.step_nums += 1
        won  = (scores_encoded == 242)
        done = won | (self.step_nums >= 6)

        rewards = np.full(self.n_envs, self.step_penalty, dtype=np.float32)

        win_done = won & done
        if np.any(win_done):
            rewards[win_done] += np.array(
                [self.win_rewards[s] for s in self.step_nums[win_done]],
                dtype=np.float32
            )

        loss_done = (~won) & done
        rewards[loss_done] += self.loss_reward

        done_indices = np.where(done)[0]
        won_indices  = np.where(win_done)[0]
        lost_indices = np.where(loss_done)[0]

        total_guesses = 0.0
        if len(won_indices)  > 0: total_guesses += self.step_nums[won_indices].sum()
        if len(lost_indices) > 0: total_guesses += len(lost_indices) * 7.0

        info = {
            "wins":        int(np.sum(win_done)),
            "dones":       int(np.sum(done)),
            "avg_guesses": total_guesses / len(done_indices) if len(done_indices) > 0 else 0.0,
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
            step_frac,
        ], axis=1)

    @property
    def training_mask(self) -> np.ndarray:
        """(vocab_size,) bool — True for words within current curriculum vocab."""
        m = np.zeros(self.vocab_size, dtype=bool)
        m[:self.vocab_limit] = True
        return m


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(net, vec_env: NumpyWordleEnv, n_games: int = 512) -> tuple[float, float]:
    """Deterministic rollout over n_games. Always uses full vocab."""
    obs, masks = vec_env.reset_all()
    wins = 0
    dones = 0
    guesses_sum = 0.0

    while dones < n_games:
        actions, _, _ = net.get_action(obs, masks, deterministic=True)
        obs, _, done, info = vec_env.step(actions)
        masks = vec_env.masks.copy()

        wins        += info["wins"]
        dones       += info["dones"]
        guesses_sum += info["avg_guesses"] * info["dones"]

    return wins / dones, guesses_sum / dones


# ══════════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--name",   type=str, default="cpu_wordle")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"Using device: {device}  (NumPy vectorised env)")

    mlflow.set_experiment("Wordle_RL_CPU")

    REWARD_CONFIG = {
        "win_rewards":  {1: 1.0, 2: 3.0, 3: 3.5, 4: 1.0, 5: 0.5, 6: -1.0},
        "loss_reward":  -5.0,
        "step_penalty": -0.1,
    }

    base_env    = WordleEnv(DATA_DIR)
    score_cache = load_score_cache(base_env)
    vec_env     = NumpyWordleEnv(base_env, score_cache, N_ENVS, reward_config=REWARD_CONFIG)
    eval_env    = NumpyWordleEnv(base_env, score_cache, N_ENVS, reward_config=REWARD_CONFIG)

    net = WordleNetwork(base_env.obs_dim, base_env.vocab_size, hidden_dim=HIDDEN_DIM).to(device)

    import sys as _sys
    if _sys.platform == "win32":
        print("torch.compile: skipped (Windows — install VS Build Tools to enable)")
    else:
        try:
            net = torch.compile(net)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, eps=1e-5, weight_decay=1e-4)

    # ── Curriculum stages ─────────────────────────────────────────────
    #  Stage │ Start iter │ Vocab  │ Approx random win rate
    # ───────┼────────────┼────────┼───────────────────────
    #    1   │     1      │    50  │  ~12%
    #    2   │   100      │   200  │  ~3%
    #    3   │   300      │   500  │  ~1.2%
    #    4   │   800      │  1500  │  ~0.4%
    #    5   │  1500      │  4000  │  ~0.15%
    #    6   │  3000      │  8000  │  ~0.075%
    #    7   │  7000      │ full   │  ~0.04%
    STAGES = [
        (1,    50),
        (100,  200),
        (300,  500),
        (800,  1500),
        (1500, 4000),
        (3000, 8000),
        (7000, 14855),
    ]

    # Fractional LR restart at each stage boundary.
    # Smaller fraction for later stages — more prior knowledge to preserve.
    LR_FRACTIONS = [1.0, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2]

    stage_lengths = []
    for i, (start, _) in enumerate(STAGES):
        end = STAGES[i + 1][0] if i + 1 < len(STAGES) else N_ITERATIONS + 1
        stage_lengths.append(end - start)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=stage_lengths[0], eta_min=1e-6
    )
    current_stage_idx = 0

    # ── Resume ────────────────────────────────────────────────────────
    start_iter = 1
    if args.resume:
        print(f"--> Resuming from: {args.resume}")
        if os.path.exists(args.resume):
            net.load_state_dict(torch.load(args.resume, map_location=device))
            m = re.search(r"(\d+)", args.resume)
            if m:
                start_iter = int(m.group(1)) + 1
        else:
            print(f"--> Checkpoint not found: {args.resume}")
            return

    run_desc = f"{args.name}_H{HIDDEN_DIM}_Ent{ENT_COEF_START}-{ENT_COEF_END}_LR{LR}"
    if args.resume:
        run_desc += f"_Res{start_iter}"
    print(f"Starting run: {run_desc}")

    with mlflow.start_run(run_name=run_desc):
        mlflow.log_params({
            "name":          args.name,
            "N_ENVS":        N_ENVS,
            "STEPS_PER_ENV": STEPS_PER_ENV,
            "BATCH":         MINIBATCH_SIZE,
            "LR":            LR,
            "HIDDEN_DIM":    HIDDEN_DIM,
            "ENT_START":     ENT_COEF_START,
            "ENT_END":       ENT_COEF_END,
            "rewards":       str(REWARD_CONFIG),
        })

        obs, _ = vec_env.reset_all()
        t0 = time.time()
        win_history = []

        for iteration in range(start_iter, N_ITERATIONS + 1):

            # Entropy: linear decay
            progress = (iteration - 1) / max(N_ITERATIONS - 1, 1)
            ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * progress

            # ── Staged curriculum ─────────────────────────────────────
            stage_idx = 0
            for i, (stage_start, _) in enumerate(STAGES):
                if iteration >= stage_start:
                    stage_idx = i
            vocab_limit = STAGES[stage_idx][1]

            if stage_idx != current_stage_idx:
                current_stage_idx = stage_idx
                new_T0     = stage_lengths[stage_idx]
                restart_lr = LR * LR_FRACTIONS[min(stage_idx, len(LR_FRACTIONS) - 1)]
                scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=new_T0, eta_min=1e-6
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = restart_lr
                print(f"  → Stage {stage_idx + 1}: vocab={vocab_limit}, "
                      f"T_0={new_T0}, restart_lr={restart_lr:.2e}")

            vec_env.vocab_limit  = vocab_limit
            eval_env.vocab_limit = vec_env.vocab_size  # eval always uses full vocab

            # Build training mask once per iteration — restricts actions to curriculum vocab
            train_mask_t     = torch.zeros(vec_env.vocab_size, dtype=torch.bool)
            train_mask_t[:vocab_limit] = True
            train_mask_batch = train_mask_t.unsqueeze(0).expand(N_ENVS, -1)

            # ── Collection ────────────────────────────────────────────
            mb_obs, mb_actions, mb_log_probs = [], [], []
            mb_values, mb_rewards, mb_dones  = [], [], []

            total_wins        = 0
            total_eps         = 0
            total_guesses_sum = 0.0

            for _ in range(STEPS_PER_ENV):
                with torch.no_grad():
                    o_t = torch.as_tensor(obs, dtype=torch.float32)
                    actions, log_probs, values = net.get_action(o_t, train_mask_batch)

                next_obs, rewards, dones, info = vec_env.step(actions)

                mb_obs.append(obs)
                mb_actions.append(actions)
                mb_log_probs.append(log_probs)
                mb_values.append(values)
                mb_rewards.append(rewards)
                mb_dones.append(dones)

                total_wins += info["wins"]
                total_eps  += info["dones"]
                if info["dones"] > 0:
                    total_guesses_sum += info["avg_guesses"] * info["dones"]

                obs = next_obs
                # masks stay all-True during training (not updated mid-episode)

            # ── PPO Update ────────────────────────────────────────────
            t_obs       = torch.tensor(np.stack(mb_obs),       dtype=torch.float32)
            t_actions   = torch.tensor(np.stack(mb_actions),   dtype=torch.long)
            t_log_probs = torch.tensor(np.stack(mb_log_probs), dtype=torch.float32)
            t_values    = torch.tensor(np.stack(mb_values),    dtype=torch.float32)
            t_rewards   = torch.tensor(np.stack(mb_rewards),   dtype=torch.float32)
            t_dones     = torch.tensor(np.stack(mb_dones),     dtype=torch.bool)

            # Bootstrap value for incomplete episodes at rollout end
            with torch.no_grad():
                _, _, last_values = net.get_action(
                    torch.as_tensor(obs, dtype=torch.float32),
                    train_mask_batch  # use current training mask, not a stale copy
                )
                last_values = torch.as_tensor(last_values, dtype=torch.float32)

            # GAE advantage estimation
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

            # Normalise advantages
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

            # Normalise returns for stable value learning
            ret_mean       = f_returns.mean()
            ret_std        = f_returns.std() + 1e-8
            f_returns_norm = (f_returns - ret_mean) / ret_std

            dataset_size = f_obs.size(0)
            epoch_pg_loss, epoch_vf_loss, epoch_entropy = [], [], []

            for _ in range(N_EPOCHS):
                # Reshuffle every epoch so each epoch sees a different minibatch order
                indices = torch.randperm(dataset_size)

                for start in range(0, dataset_size, MINIBATCH_SIZE):
                    idx = indices[start: start + MINIBATCH_SIZE]

                    # mask=None during update: actions already sampled,
                    # we just need unmasked logits for the ratio computation
                    logits, values = net(f_obs[idx], mask=None)
                    dist    = Categorical(logits=logits)
                    new_lp  = dist.log_prob(f_actions[idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - f_log_probs[idx])
                    surr1 = ratio * f_adv[idx]
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * f_adv[idx]
                    pg_loss = -torch.min(surr1, surr2).mean()

                    # Value loss with clipping
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

            metrics = {
                "win_rate":     avg_win_rate,
                "avg_guesses":  avg_guesses,
                "eps_per_iter": total_eps,
                "pg_loss":      np.mean(epoch_pg_loss),
                "vf_loss":      np.mean(epoch_vf_loss),
                "entropy":      np.mean(epoch_entropy),
                "ent_coef":     ent_coef,
                "lr":           scheduler.get_last_lr()[0],
                "vocab_limit":  vocab_limit,
            }
            mlflow.log_metrics(metrics, step=iteration)

            if iteration % LOG_EVERY == 0:
                smooth_win = np.mean(win_history[-10:]) if win_history else 0.0
                elapsed    = time.time() - t0
                print(
                    f"Iter {iteration:5d} | "
                    f"Win: {smooth_win:.2%} | "
                    f"Guess: {avg_guesses:.2f} | "
                    f"Ent: {metrics['entropy']:.3f} | "
                    f"Vocab: {vocab_limit}/{vec_env.vocab_size} | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"t: {elapsed:.0f}s"
                )

            if iteration % 250 == 0:
                eval_win, eval_guess = evaluate(net, eval_env)
                mlflow.log_metrics(
                    {"eval_win_rate": eval_win, "eval_avg_guesses": eval_guess},
                    step=iteration
                )
                print(f"  → EVAL: Win {eval_win:.2%}  Avg guesses {eval_guess:.3f}")

            if iteration % SAVE_EVERY == 0:
                path  = f"{MODEL_DIR}/{args.name}_{iteration}.pt"
                state = getattr(net, "_orig_mod", net).state_dict()
                torch.save(state, path)
                mlflow.log_artifact(path)

    final_path = f"{MODEL_DIR}/{args.name}_final.pt"
    state = getattr(net, "_orig_mod", net).state_dict()
    torch.save(state, final_path)
    mlflow.log_artifact(final_path)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()