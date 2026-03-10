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

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
N_ENVS         = 64         # Decreased parallel games to offset horizon
STEPS_PER_ENV  = 128        # Increased horizon (Batch size = 8192)
MINIBATCH_SIZE = 2048       # Size of SGD chunks
N_EPOCHS       = 4          # PPO epochs per update
N_ITERATIONS   = 50000      # Total training iterations

LR             = 3e-4       # Starting learning rate (will decay)
MIN_LR         = 1e-5       # NEW: The absolute lowest the LR can go
GAMMA          = 0.99       # Discount factor
GAE_LAMBDA     = 0.95       # GAE smoothing parameter
CLIP_EPS       = 0.2        # PPO clip range
ENT_COEF       = 0.1       # Starting exploration bonus (will decay)
VF_COEF        = 0.5        # Value loss weight
INFO_COEF      = 0.1        # Information Gain reward weight
MAX_GRAD_NORM  = 0.5        # Gradient clipping

LOG_EVERY = 1

DATA_DIR       = "data"
MODEL_DIR      = "models"

CURRICULUM_PHASE = 1           # 1 = Only guess candidate secrets (2.3k), 2 = All words (14k)
LOAD_CHECKPOINT  = False       # Set to True when moving to Phase 2 (or resuming)
CHECKPOINT_PATH  = "models/wordle_it1000.pt"
# ──────────────────────────────────────────────────────────────────────────────

class NumpyWordleEnv:
    """Fully vectorized Wordle logic for high-speed CPU training."""
    def __init__(self, base_env, score_cache_np, n_envs, test_indices=None):
        self.n_envs = n_envs
        self.vocab_size = base_env.vocab_size
        self.test_indices = test_indices
        self.score_cache = score_cache_np
        
        # Pre-decoded color cache (Vocab, Vocab, 5)
        self.color_cache = np.zeros((self.vocab_size, self.vocab_size, 5), dtype=np.uint8)
        temp = score_cache_np.copy()
        for i in range(5):
            self.color_cache[:, :, i] = temp % 3
            temp //= 3

        self.words_int = np.array([[ord(c)-ord('a') for c in w] for w in base_env.words])
        self.reset_all()

    def _sample_secrets(self, size):
        pool = self.test_indices if self.test_indices is not None else self.vocab_size
        return np.random.choice(pool, size)

    def reset_all(self):
        self.secret_idxs = self._sample_secrets(self.n_envs)
        self.step_nums = np.zeros(self.n_envs, dtype=np.int32)
        self.masks = np.ones((self.n_envs, self.vocab_size), dtype=bool)
        self.min_counts = np.zeros((self.n_envs, 26), dtype=np.float32)
        self.max_counts = np.ones((self.n_envs, 26), dtype=np.float32) * 5.0
        self.letter_green = np.zeros((self.n_envs, 5, 26), dtype=np.float32)
        self.letter_yellow_not = np.zeros((self.n_envs, 5, 26), dtype=np.float32)
        return self._get_obs()

    def step(self, actions):
        current_colors = self.color_cache[actions, self.secret_idxs]
        guess_letters = self.words_int[actions]
        encoded_scores = self.score_cache[actions, self.secret_idxs]
        
        # --- 1. Check validity BEFORE mutating the masks ---
        was_valid = self.masks[np.arange(self.n_envs), actions].copy()
        # ---------------------------------------------------

        # Calculate Information Gain
        words_before = self.masks.sum(axis=1)
        for i in range(self.n_envs):
            self.masks[i] &= (self.score_cache[actions[i]] == encoded_scores[i])
        words_after = self.masks.sum(axis=1)
        info_gain = np.log2(words_before + 1) - np.log2(words_after + 1)
        
        # Update Observation state
        env_ids = np.arange(self.n_envs)
        for p in range(5):
            l, c = guess_letters[:, p], current_colors[:, p]
            self.letter_yellow_not[env_ids[c==1], p, l[c==1]] = 1.0
            self.letter_green[env_ids[c==2], p, l[c==2]] = 1.0

        for char_code in range(26):
            l_mask = (guess_letters == char_code)
            total_g = l_mask.sum(axis=1)
            colored_g = (l_mask & (current_colors > 0)).sum(axis=1)
            self.min_counts[:, char_code] = np.maximum(self.min_counts[:, char_code], colored_g)
            ceiling_found = total_g > colored_g
            self.max_counts[ceiling_found, char_code] = colored_g[ceiling_found]

        self.step_nums += 1
        won = (encoded_scores == 242)
        done = won | (self.step_nums >= 6)
        
        # --- 2. THE NEW REWARD LOGIC ---
        # Base step penalty
        rewards = np.full(self.n_envs, -0.05, dtype=np.float32) 
        
        # Add info gain (only if it was a valid guess, to prevent hacking)
        rewards += (info_gain * INFO_COEF) * was_valid 
        
        # Penalize guessing words we already know are wrong
        rewards[~was_valid] -= 0.5 
        
        rewards[won] += 2.0
        rewards[(~won) & done] -= 5.0
        # -------------------------------

        # --- 3. EXTRACT GUESS COUNTS FOR COMPLETED GAMES ---
        guesses_for_dones = self.step_nums[done].copy()
        # ---------------------------------------------------

        # (Notice the old rewards block that was here is now deleted!)

        # --- 4. ADD TO INFO DICT ---
        info = {
            "wins": won.sum(), 
            "dones": done.sum(),
            "guesses": guesses_for_dones
        }
        # ---------------------------
        
        if done.any():
            self._reset_specific(np.where(done)[0])
            
        return self._get_obs(), rewards, done, info

    def _reset_specific(self, idxs):
        self.secret_idxs[idxs] = self._sample_secrets(len(idxs))
        self.step_nums[idxs] = 0
        self.masks[idxs] = True
        self.min_counts[idxs] = 0
        self.max_counts[idxs] = 5.0
        self.letter_green[idxs] = 0
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
    parser.add_argument("--load", action="store_true", 
                        help="Flag to load a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the checkpoint file to load")
    parser.add_argument("--start-iter", type=int, default=1,
                        help="The iteration number to start counting from (useful for resuming)")
    parser.add_argument("--iters", type=int, default=5000, 
                        help="Number of iterations to run THIS session")
    
    args = parser.parse_args()


    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cpu")
    
    base_env = WordleEnv(DATA_DIR)
    score_cache = np.load(os.path.join(DATA_DIR, "score_cache.npy"))
    vec_env = NumpyWordleEnv(base_env, score_cache, N_ENVS)
    
    net = WordleNetwork(base_env.OBS_DIM, base_env.vocab_size).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, eps=1e-5)

    # --- 1. CHECKPOINT LOADING ---
    if args.load:
        load_path = args.checkpoint
        
        # If no explicit checkpoint was provided, find the highest one for this phase
        if load_path is None:
            import glob
            import re
            
            # Find all files matching this phase (e.g., wordle_phase1_it*.pt)
            pattern = os.path.join(MODEL_DIR, f"wordle_phase{args.phase}_it*.pt")
            checkpoints = glob.glob(pattern)
            
            if checkpoints:
                # Extract the integer 'X' from '_itX.pt' and find the max
                def get_iter(filepath):
                    match = re.search(r"_it(\d+)\.pt$", filepath)
                    return int(match.group(1)) if match else -1
                
                load_path = max(checkpoints, key=get_iter)
            else:
                load_path = f"{MODEL_DIR}/wordle_phase{args.phase}_it0.pt" # Fallback that won't exist

        if os.path.exists(load_path):
            print(f" [INFO] Loading checkpoint from {load_path}...")
            net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            print(f" [WARNING] Checkpoint {load_path} not found. Starting fresh.")

    if args.phase == 1:
        print(" [INFO] CURRICULUM PHASE 1: Restricting actions to the candidate secret words (2.3k).")
        curriculum_mask = torch.zeros(base_env.vocab_size, dtype=torch.bool)
        curriculum_mask[base_env.test_indices] = True
        curriculum_mask = curriculum_mask.to(device)
    else:
        print(" [INFO] CURRICULUM PHASE 2: Full 14,000 action space unlocked.")
        curriculum_mask = None

    sqlite_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_path}")

    mlflow.set_experiment("Wordle_RL_CPU")
    with mlflow.start_run():
        mlflow.log_params({
            "lr": LR, "n_envs": N_ENVS, "steps": STEPS_PER_ENV, 
            "info_coef": INFO_COEF, "batch_size": N_ENVS * STEPS_PER_ENV,
            "phase": args.phase  # <--- NEW: Log the phase
        })
        
        obs = vec_env.reset_all()
        start_time = time.time()

        # --- 3. NEW METRIC TRACKERS ---
        log_wins = 0
        log_dones = 0
        log_guesses = []

        for iteration in range(args.start_iter, args.start_iter + args.iters):
            
            # --- Linear Decay Scheduler ---
            frac = 1.0 - (iteration - 1.0) / N_ITERATIONS
            current_lr = MIN_LR + (LR - MIN_LR) * frac
            current_ent = max(0.001, ENT_COEF * frac)
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            # ------------------------------

            # ─── 1. TRAJECTORY COLLECTION ───
            mb_obs, mb_actions, mb_log_probs, mb_rewards, mb_values, mb_dones = [], [], [], [], [], []

            for _ in range(STEPS_PER_ENV):
                with torch.no_grad():
                    o_t = torch.as_tensor(obs)
                    # mask=None explicitly disables Hard Mode during acting
                    actions, log_probs, values = net.get_action(o_t, mask=curriculum_mask)

                next_obs, rewards, dones, info = vec_env.step(actions.numpy())
                
                # --- 4. ACCUMULATE METRICS ---
                log_wins += info["wins"]
                log_dones += info["dones"]
                log_guesses.extend(info["guesses"].tolist())
                # -----------------------------

                mb_obs.append(o_t)
                mb_actions.append(actions)
                mb_log_probs.append(log_probs)
                mb_rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
                mb_values.append(values.flatten())
                mb_dones.append(torch.as_tensor(dones))
                obs = next_obs

            # ─── 2. ADVANTAGE CALCULATION (GAE) ───
            with torch.no_grad():
                _, _, last_value = net.get_action(torch.as_tensor(obs), mask=None)
                last_value = last_value.flatten()

            mb_advantages = torch.zeros_like(torch.stack(mb_rewards))
            last_gae = 0
            for t in reversed(range(STEPS_PER_ENV)):
                if t == STEPS_PER_ENV - 1:
                    next_non_terminal = 1.0 - mb_dones[t].float()
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - mb_dones[t].float()
                    next_value = mb_values[t+1]
                
                delta = mb_rewards[t] + GAMMA * next_value * next_non_terminal - mb_values[t]
                mb_advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
            
            mb_returns = mb_advantages + torch.stack(mb_values)

            # Flatten mini-batches
            b_obs = torch.cat(mb_obs)
            b_actions = torch.cat(mb_actions)
            b_log_probs = torch.cat(mb_log_probs)
            b_advantages = mb_advantages.flatten()
            
            # --- ADVANTAGE NORMALIZATION ---
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            # -------------------------------
            
            b_returns = mb_returns.flatten()

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
            if iteration % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                avg_rew = torch.stack(mb_rewards).mean().item()

                actual_entropy = np.mean(mb_entropies)
                actual_vf_loss = np.mean(mb_vf_losses) # <--- NEW
                
                # --- 5. CALCULATE HUMAN METRICS ---
                win_rate = (log_wins / log_dones) if log_dones > 0 else 0.0
                avg_guesses = np.mean(log_guesses) if len(log_guesses) > 0 else 6.0
                # ----------------------------------

                # Updated print statement
                print(f"Iter {iteration:4d} | Rew: {avg_rew:+.3f} | Win: {win_rate*100:5.1f}% | Guess: {avg_guesses:.2f} | LR: {current_lr:.5f} | Ent: {actual_entropy:.4f} | Time: {elapsed:.0f}s")
                
                mlflow.log_metric("avg_reward", avg_rew, step=iteration)
                mlflow.log_metric("entropy_coef", current_ent, step=iteration)

                mlflow.log_metric("policy_entropy", actual_entropy, step=iteration)
                mlflow.log_metric("value_loss", actual_vf_loss, step=iteration) # <--- NEW
                
                # --- 6. LOG TO MLFLOW & RESET ---
                mlflow.log_metric("win_rate", win_rate, step=iteration)
                mlflow.log_metric("avg_guesses", avg_guesses, step=iteration)
                
                log_wins = 0
                log_dones = 0
                log_guesses = []
                # --------------------------------

            if iteration % 100 == 0:
                # Dynamically insert the phase number into the filename
                save_path = f"{MODEL_DIR}/wordle_phase{args.phase}_it{iteration}.pt"
                torch.save(net.state_dict(), save_path)

if __name__ == "__main__":
    main()