import sys
import os
import argparse
import math

# --- PATH SETUP ---
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
# ------------------

from collections import deque
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import mlflow

from env.wordle_env import WordleEnv
from agent.network import WordleNetwork

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
N_ENVS         = 64 # Decreased parallel games to offset horizon
STEPS_PER_ENV  = 128        # Increased horizon (Batch size = 8192)
MINIBATCH_SIZE = N_ENVS * STEPS_PER_ENV // 6  # Size of SGD chunks
N_EPOCHS       = 4          # PPO epochs per update
N_ITERATIONS   = 30000      # Total training iterations
N_DIMS = 1024

LR             = 5e-5       # Starting learning rate (will decay)
MIN_LR         = 1e-5       # NEW: The absolute lowest the LR can go
GAMMA          = 0.99       # Discount factor
GAE_LAMBDA     = 0.95       # GAE smoothing parameter
CLIP_EPS       = 0.2        # PPO clip range
ENT_COEF       = 0.1        # Starting exploration bonus (will decay)
VF_COEF        = 0.5        # Value loss weight

INFO_COEF_START = 0.5
INFO_COEF_END   = 0.05       # Information Gain reward weight

MAX_GRAD_NORM  = 0.5        # Gradient clipping

LOG_EVERY = 1
SAVE_FREQ = 500

DATA_DIR       = "data"
MODEL_DIR      = "models"

CURRICULUM_PHASE = 1           # 1 = Only guess candidate secrets (2.3k), 2 = All words (14k)
LOAD_CHECKPOINT  = False       # Set to True when moving to Phase 2 (or resuming)
# ──────────────────────────────────────────────────────────────────────────────

class NumpyWordleEnv:
    def __init__(self, base_env, score_cache_np, n_envs):
        self.n_envs      = n_envs
        self.vocab_size  = base_env.vocab_size
        self.n_secrets   = score_cache_np.shape[1]
        self.score_cache = score_cache_np               # (vocab_size x n_secrets)

        # Maps vocab index -> secret index (-1 if not a candidate secret)
        self.vocab_to_secret = np.full(self.vocab_size, -1, dtype=np.int32)
        for s_idx, v_idx in enumerate(base_env.test_indices):
            self.vocab_to_secret[v_idx] = s_idx

        self.color_cache = np.zeros((self.vocab_size, self.n_secrets, 5), dtype=np.uint8)
        temp = score_cache_np.copy()
        for i in range(5):
            self.color_cache[:, :, i] = temp % 3
            temp //= 3

        self.words_int = np.array(
            [[ord(c) - ord('a') for c in w] for w in base_env.words]
        )
        self.reset_all()

    def _sample_secrets(self, size):
        return np.random.randint(0, self.n_secrets, size=size)

    def reset_all(self):
        self.secret_idxs       = self._sample_secrets(self.n_envs)
        self.step_nums         = np.zeros(self.n_envs, dtype=np.int32)
        self.masks             = np.ones((self.n_envs, self.n_secrets), dtype=bool)  # now n_secrets wide
        self.min_counts        = np.zeros((self.n_envs, 26), dtype=np.float32)
        self.max_counts        = np.ones((self.n_envs, 26), dtype=np.float32) * 5.0
        self.letter_green      = np.zeros((self.n_envs, 5, 26), dtype=np.float32)
        self.letter_yellow_not = np.zeros((self.n_envs, 5, 26), dtype=np.float32)
        return self._get_obs()

    def step(self, actions, info_coef):
        current_colors = self.color_cache[actions, self.secret_idxs]
        guess_letters  = self.words_int[actions]
        encoded_scores = self.score_cache[actions, self.secret_idxs]

        # Information gain over n_secrets axis
        words_before = self.masks.sum(axis=1)
        
        action_scores = self.score_cache[actions]  # (n_envs, n_secrets)
        encoded_scores_expanded = encoded_scores[:, None]  # (n_envs, 1)
        
        self.masks &= (action_scores == encoded_scores_expanded)
        
        words_after = self.masks.sum(axis=1)
        info_gain = np.log2(words_before + 1) - np.log2(words_after + 1)  # keep for logging
        progress_reward = np.log2(self.n_secrets + 1) - np.log2(words_after + 1)
        
        # Update Observation state
        env_ids = np.arange(self.n_envs)
        for p in range(5):
            l, c = guess_letters[:, p], current_colors[:, p]
            self.letter_yellow_not[env_ids[c==1], p, l[c==1]] = 1.0
            self.letter_green[env_ids[c==2], p, l[c==2]] = 1.0

        
        char_one_hot = (guess_letters[:, :, None] == np.arange(26))  # (n_envs, 5, 26)
        colored = (current_colors > 0)[:, :, None]                   # (n_envs, 5, 1)

        total_g   = char_one_hot.sum(axis=1)                         # (n_envs, 26)
        colored_g = (char_one_hot & colored).sum(axis=1)             # (n_envs, 26)

        self.min_counts = np.maximum(self.min_counts, colored_g)
        ceiling_found = total_g > colored_g
        self.max_counts = np.where(ceiling_found, colored_g, self.max_counts)

        self.step_nums += 1
        won = (encoded_scores == 242)
        done = won | (self.step_nums >= 6)
        
        # --- 2. THE NEW REWARD LOGIC ---
        # Base step penalty
        rewards = np.full(self.n_envs, -0.3, dtype=np.float32)
        rewards += info_gain * info_coef
        guesses_remaining = 6 - self.step_nums  # ← add this
        rewards[won] += 8.0 + (guesses_remaining[won] * 1)
        rewards[(~won) & done] -= 2.0
        # -------------------------------

        # --- 3. EXTRACT GUESS COUNTS FOR COMPLETED GAMES ---
        guesses_for_dones = self.step_nums[done].copy()
        # ---------------------------------------------------

        # (Notice the old rewards block that was here is now deleted!)

        # --- 4. ADD TO INFO DICT ---
        info = {
            "wins": won.sum(), 
            "dones": done.sum(),
            "guesses": guesses_for_dones,
            "avg_info_gain" : info_gain.mean(),
            "avg_progress": progress_reward.mean()
        }
        
        # ---------------------------
        
        if done.any():
            self._reset_specific(np.where(done)[0])
            
        return self._get_obs(), rewards, done, info

    def _reset_specific(self, idxs):
        self.secret_idxs[idxs]       = self._sample_secrets(len(idxs))
        self.step_nums[idxs]         = 0
        self.masks[idxs]             = True   # still works, now resets (len(idxs), n_secrets) rows
        self.min_counts[idxs]        = 0
        self.max_counts[idxs]        = 5.0
        self.letter_green[idxs]      = 0
        self.letter_yellow_not[idxs] = 0

    def _get_obs(self):
        return np.concatenate([
            self.min_counts / 5.0, self.max_counts / 5.0,
            self.letter_green.reshape(self.n_envs, -1),
            self.letter_yellow_not.reshape(self.n_envs, -1),
            (self.step_nums / 6.0)[:, None]
        ], axis=1).astype(np.float32)


def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Wordle PPO Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], 
                        help="Curriculum Phase: 1 (Answers only) or 2 (All 14k words)")
    parser.add_argument("--load", type=str, default=None, 
                        help="Path to the checkpoint file to load")
    parser.add_argument("--start-iter", type=int, default=1,
                        help="The iteration number to start counting from (useful for resuming)")
    parser.add_argument("--iters", type=int, default=5000, 
                        help="Number of iterations to run THIS session")
    parser.add_argument("--name", type=str, default="wordle",
                        help="Base name for saved models (e.g., 'wordle')")
    parser.add_argument("--dims", type=int, default=1024,
                        help="Number of hidden dimensions of MLP")
    
    args = parser.parse_args()


    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        global N_ENVS, STEPS_PER_ENV, MINIBATCH_SIZE, N_EPOCHS, N_ITERATIONS
        N_ENVS         = 1024       #  
        STEPS_PER_ENV  = 128        # 
        MINIBATCH_SIZE = 8192       # Size of SGD chunks
        N_EPOCHS       = 2          # PPO epochs per update
        N_ITERATIONS   = 1000       # Total training iterations
        SAVE_FREQ      = 100
    
    base_env = WordleEnv(DATA_DIR)
    score_cache = np.load(os.path.join(DATA_DIR, "score_cache.npy"))
    vec_env = NumpyWordleEnv(base_env, score_cache, N_ENVS)
    
    net = WordleNetwork(base_env.OBS_DIM, base_env.vocab_size, hidden_dim=args.dims).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, eps=1e-5)

    # --- 1. CHECKPOINT LOADING ---
    if args.load:
        load_path = args.load
        if os.path.exists(load_path):
            print(f" [INFO] Loading checkpoint from {load_path}...")
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f" [INFO] Optimizer state restored.")
            else:
                # backwards compatibility with old checkpoints that only saved model
                net.load_state_dict(checkpoint)
                print(f" [INFO] Old checkpoint format - optimizer state not restored.")

    if args.phase == 1:
        print(" [INFO] CURRICULUM PHASE 1: Restricting actions to the candidate secret words.")
        curriculum_mask = torch.zeros(base_env.vocab_size, dtype=torch.bool)
        curriculum_mask[base_env.test_indices] = True
        curriculum_mask = curriculum_mask.to(device)
    else:
        print(" [INFO] CURRICULUM PHASE 2: Full action space unlocked.")
        curriculum_mask = None

    sqlite_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_path}")

    mlflow.set_experiment("Wordle_RL_CPU")
    with mlflow.start_run():
        mlflow.log_params({
            "decay iters": N_ITERATIONS,
            "lr": LR, 
            "end_lr": MIN_LR,
            "n_envs": N_ENVS, 
            "steps": STEPS_PER_ENV, 
            "info_coef start": INFO_COEF_START,
            "info_coef_end": INFO_COEF_END, 
            "entropy_coef start": ENT_COEF,
            "batch_size": N_ENVS * STEPS_PER_ENV,
            "phase": args.phase 
        })
        
        print(f"lr: {LR} \
              \nn_envs: {N_ENVS} \
              \nsteps: {STEPS_PER_ENV} \
              \ninfo_coef: {current_info_coef} \
              \nbatchsize: {N_ENVS * STEPS_PER_ENV} \
              \nphase: {args.phase} \
              \nentropy_coef: {ENT_COEF}")
        
        obs = vec_env.reset_all()
        start_time = time.time()

        # --- 3. NEW METRIC TRACKERS ---
        log_wins = 0
        log_dones = 0
        log_guesses = []
        log_info_gains = []  # ← add this
        log_progress = []
        obs_dim = base_env.OBS_DIM
        buf_obs      = torch.zeros(STEPS_PER_ENV, N_ENVS, obs_dim).to(device)
        buf_actions  = torch.zeros(STEPS_PER_ENV, N_ENVS, dtype=torch.long).to(device)
        buf_logprobs = torch.zeros(STEPS_PER_ENV, N_ENVS).to(device)
        buf_rewards  = torch.zeros(STEPS_PER_ENV, N_ENVS).to(device)
        buf_values   = torch.zeros(STEPS_PER_ENV, N_ENVS).to(device)
        buf_dones    = torch.zeros(STEPS_PER_ENV, N_ENVS, dtype=torch.bool).to(device)
        win_rate_window = deque(maxlen=5000)  # last 1000 completed games
        
        for iteration in range(args.start_iter, args.start_iter + args.iters):
            # --- Linear Decay Scheduler ---
            frac = 1.0 - (iteration - 1.0) / N_ITERATIONS
            frac = max(0, frac)
            current_lr = MIN_LR + (LR - MIN_LR) * frac
            current_ent = max(0.0001, ENT_COEF * frac)
            current_info_coef = INFO_COEF_END + (INFO_COEF_START - INFO_COEF_END) * frac
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            # ------------------------------

            # ─── 1. TRAJECTORY COLLECTION ───

            for step in range(STEPS_PER_ENV):                
                with torch.inference_mode():
                    o_t = torch.as_tensor(obs).to(device)
                    actions, log_probs, values = net.get_action(o_t, mask=curriculum_mask)
                next_obs, rewards, dones, info = vec_env.step(actions.cpu().numpy(), current_info_coef)

                # --- ACCUMULATE METRICS (unchanged) ---
                log_wins += info["wins"]
                log_dones += info["dones"]
                log_guesses.extend(info["guesses"].tolist())
                log_info_gains.append(info["avg_info_gain"])
                log_progress.append(info["avg_progress"])
                
                win_rate_window.extend([1] * info["wins"])
                win_rate_window.extend([0] * (info["dones"] - info["wins"]))
                # --------------------------------------

                # --- BUFFER WRITES (replaces list appends) ---
                buf_obs[step]      = o_t
                buf_actions[step]  = actions
                buf_logprobs[step] = log_probs
                buf_rewards[step] = torch.as_tensor(rewards, dtype=torch.float32).to(device)
                buf_dones[step]   = torch.as_tensor(dones).to(device)
                buf_values[step]   = values.flatten()
                obs = next_obs

            # ─── 2. ADVANTAGE CALCULATION (GAE) ───
            with torch.inference_mode():
                _, _, last_value = net.get_action(torch.as_tensor(obs).to(device), mask=curriculum_mask)
                last_value = last_value.flatten()

            mb_advantages = torch.zeros(STEPS_PER_ENV, N_ENVS).to(device)
            last_gae = 0
            for t in reversed(range(STEPS_PER_ENV)):
                if t == STEPS_PER_ENV - 1:
                    next_non_terminal = 1.0 - buf_dones[t].float()
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - buf_dones[t].float()
                    next_value = buf_values[t+1]
                
                delta = buf_rewards[t] + GAMMA * next_value * next_non_terminal - buf_values[t]
                mb_advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae

            mb_returns = mb_advantages + buf_values

            # Flatten
            b_obs        = buf_obs.flatten(0, 1)
            b_actions    = buf_actions.flatten(0, 1)
            b_log_probs  = buf_logprobs.flatten(0, 1)
            b_advantages = mb_advantages.flatten()
            b_returns    = mb_returns.flatten()

            # --- ADVANTAGE NORMALIZATION ---
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # ─── 3. PPO OPTIMIZATION ───            
            b_inds = np.arange(N_ENVS * STEPS_PER_ENV)
            mb_entropies = []
            mb_vf_losses = []
            for epoch in range(N_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_inds), MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    # Note: We pass mask=None here as well to match trajectory logic
                    new_logits, new_values = net(b_obs[mb_inds], mask=curriculum_mask)
                    new_dist = Categorical(logits=new_logits)
                    new_log_probs = new_dist.log_prob(b_actions[mb_inds])
                    
                    # Policy Loss
                    ratio = torch.exp(new_log_probs - b_log_probs[mb_inds])
                    surr1 = ratio * b_advantages[mb_inds]
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages[mb_inds]
                    pg_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss
                    vf_loss = 0.5 * F.mse_loss(new_values.flatten(), b_returns[mb_inds])

                    # Entropy Loss
                    entropy_loss = new_dist.entropy().mean()

                    mb_entropies.append(entropy_loss.item())
                    mb_vf_losses.append(vf_loss.item()) # <--- NEW

                    # Loss Calculation using current_ent
                    loss = pg_loss + VF_COEF * vf_loss - current_ent * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            # ─── 4. LOGGING ───
            if (iteration - 1) % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_rew = buf_rewards.mean().item()

                actual_entropy = np.mean(mb_entropies)
                actual_vf_loss = np.mean(mb_vf_losses) # <--- NEW
                
                # --- 5. CALCULATE HUMAN METRICS ---
                win_rate = (log_wins / log_dones) if log_dones > 0 else 0.0
                avg_guesses = np.mean(log_guesses) if len(log_guesses) > 0 else 6.0
                
                avg_info_gain = np.mean(log_info_gains) if len(log_info_gains) > 0 else 0.0
                avg_progress = np.mean(log_progress) if len(log_progress) > 0 else 0.0
                
                rolling_win_rate = np.mean(win_rate_window) if len(win_rate_window) > 0 else 0.0
                # ----------------------------------

                # Updated print statement
                print(f"Iter {iteration:4d} | Rew: {avg_rew:+.3f} | Win: {win_rate*100:5.1f}% | Guess: {avg_guesses:.2f} | LR: {current_lr:.5f} | Ent: {actual_entropy:.4f} | Time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
                
                mlflow.log_metric("avg_reward", avg_rew, step=iteration)
                mlflow.log_metric("entropy_coef", current_ent, step=iteration)
                mlflow.log_metric("learning_rate", current_lr, step=iteration)

                mlflow.log_metric("policy_entropy", actual_entropy, step=iteration)
                mlflow.log_metric("value_loss", actual_vf_loss, step=iteration) # <--- NEW
                
                # --- 6. LOG TO MLFLOW & RESET ---
                mlflow.log_metric("win_rate", win_rate, step=iteration)
                mlflow.log_metric("avg_guesses", avg_guesses, step=iteration)
                mlflow.log_metric("avg_info_gain", avg_info_gain, step=iteration)
                mlflow.log_metric("avg_progress", avg_progress, step=iteration)
                mlflow.log_metric("rolling_win_rate", rolling_win_rate, step=iteration)
                mlflow.log_metric("info_coef", current_info_coef, step=iteration)
                
                log_wins = 0
                log_dones = 0
                log_guesses = []
                log_info_gains = []
                log_progress = []
                
                # --------------------------------

            if iteration % SAVE_FREQ == 0:
                # Dynamically insert the phase number into the filename
                save_path = f"{MODEL_DIR}/{args.name}_phase{args.phase}_it{iteration}.pt"
                # SAVING - replace current save with this
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration
                }, save_path)
                print(f" [SAVED] Checkpoint successfully written to {save_path}")
            if iteration % 100 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved  = torch.cuda.memory_reserved() / 1e9
                print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

if __name__ == "__main__":
    main()