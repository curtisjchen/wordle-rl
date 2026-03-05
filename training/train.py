"""
train.py — Main training loop.

Usage:
    python training/train.py

The script logs win-rate and loss every LOG_EVERY iterations and saves
model checkpoints under models/.

Expected convergence (CPU, default settings):
    ~1000 iterations  →  ~70 % win rate
    ~3000 iterations  →  ~85 % win rate
    ~5000 iterations  →  ~90 %+ win rate
"""

import sys
import os
# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch

from env.wordle_env  import WordleEnv
from agent.network   import WordleNetwork
from agent.ppo       import PPOTrainer


# ── Hyperparameters ───────────────────────────────────────────────────────────
DATA_DIR          = "data"
MODEL_DIR         = "models"

EPISODES_PER_ITER = 128       # episodes collected before each update
N_ITERATIONS      = 5_000     # total training iterations
LR                = 3e-4
HIDDEN_DIM        = 256

LOG_EVERY         = 25        # print stats every N iterations
SAVE_EVERY        = 500       # save checkpoint every N iterations
EVAL_EPISODES     = 500       # episodes for periodic evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(env, network, n_episodes=500):
    """Greedy evaluation — returns win rate and average guesses."""
    wins, total_guesses = 0, 0
    for _ in range(n_episodes):
        obs, mask = env.reset()
        done = False
        while not done:
            action, _, _ = network.get_action(obs, mask, deterministic=True)
            obs, _, done, info = env.step(action)
            mask = env.valid_mask()
        if info["won"]:
            wins += 1
        total_guesses += env.step_num
    return wins / n_episodes, total_guesses / n_episodes


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Environment ──────────────────────────────────────────────────────────
    env = WordleEnv(DATA_DIR)

    # ── Network ──────────────────────────────────────────────────────────────
    net = WordleNetwork(
        obs_dim    = env.obs_dim,
        vocab_size = env.vocab_size,
        hidden_dim = HIDDEN_DIM,
    )
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n{'='*60}")
    print(f"  Wordle RL — PPO Training")
    print(f"{'='*60}")
    print(f"  Network parameters : {total_params:,}")
    print(f"  Vocabulary size    : {env.vocab_size:,}")
    print(f"  Observation dim    : {env.obs_dim}")
    print(f"  Episodes / iter    : {EPISODES_PER_ITER}")
    print(f"  Total iterations   : {N_ITERATIONS:,}")
    print(f"{'='*60}\n")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = PPOTrainer(
        network    = net,
        lr         = LR,
        clip_eps   = 0.2,
        vf_coef    = 0.5,
        ent_coef   = 0.01,
        gamma      = 0.99,
        gae_lambda = 0.95,
        n_epochs   = 4,
        batch_size = 256,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    win_history = []
    t_start     = time.time()

    for iteration in range(1, N_ITERATIONS + 1):
        buffer, stats = trainer.collect_rollouts(env, EPISODES_PER_ITER)
        update_stats  = trainer.update(buffer)

        win_history.append(stats["win_rate"])

        # ── Logging ──────────────────────────────────────────────────────────
        if iteration % LOG_EVERY == 0:
            recent_win = np.mean(win_history[-LOG_EVERY:])
            elapsed    = time.time() - t_start
            print(
                f"  iter {iteration:5d}/{N_ITERATIONS} | "
                f"win {recent_win:5.1%} | "
                f"avg_guesses {stats['avg_guesses']:.2f} | "
                f"pg {update_stats['pg_loss']:+.4f} | "
                f"vf {update_stats['vf_loss']:.4f} | "
                f"ent {update_stats['entropy']:.3f} | "
                f"{elapsed/60:.1f}m"
            )

        # ── Periodic greedy eval ──────────────────────────────────────────────
        if iteration % (LOG_EVERY * 10) == 0:
            wr, ag = evaluate(env, net, EVAL_EPISODES)
            print(f"  {'─'*52}")
            print(f"  [EVAL]  greedy win rate = {wr:.1%}  avg guesses = {ag:.2f}")
            print(f"  {'─'*52}")

        # ── Checkpointing ─────────────────────────────────────────────────────
        if iteration % SAVE_EVERY == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_{iteration:05d}.pt")
            torch.save(
                {
                    "iteration":  iteration,
                    "state_dict": net.state_dict(),
                    "win_rate":   np.mean(win_history[-LOG_EVERY:]),
                },
                ckpt_path,
            )
            print(f"  → checkpoint saved: {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, "final.pt")
    torch.save({"state_dict": net.state_dict()}, final_path)

    wr, ag = evaluate(env, net, EVAL_EPISODES)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Final greedy win rate : {wr:.1%}")
    print(f"  Final avg guesses     : {ag:.2f}")
    print(f"  Model saved to        : {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()