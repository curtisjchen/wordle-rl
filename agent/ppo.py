"""
ppo.py — Proximal Policy Optimisation from scratch.

Key components
--------------
RolloutBuffer  : stores transitions for one iteration, computes GAE
PPOTrainer     : collects rollouts and performs the PPO update
"""

import numpy as np
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
class RolloutBuffer:
    """
    Stores one iteration of experience.

    Each timestep stores:
        obs        (obs_dim,)
        action     scalar int
        reward     scalar float
        done       scalar bool
        log_prob   scalar float   — from the behaviour policy
        value      scalar float   — V(s) estimate
        valid_mask (vocab_size,)  — action mask at this timestep
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs        = []
        self.actions    = []
        self.rewards    = []
        self.dones      = []
        self.log_probs  = []
        self.values     = []
        self.valid_masks = []

    def add(self, obs, action, reward, done, log_prob, value, valid_mask):
        self.obs        .append(obs)
        self.actions    .append(action)
        self.rewards    .append(float(reward))
        self.dones      .append(float(done))
        self.log_probs  .append(float(log_prob))
        self.values     .append(float(value))
        self.valid_masks.append(valid_mask)

    def __len__(self):
        return len(self.rewards)

    # ---------------------------------------------------------------- GAE
    def compute_gae(
        self,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ):
        """
        Generalised Advantage Estimation.
        Returns (advantages, returns) as float32 numpy arrays.
        """
        n       = len(self.rewards)
        adv     = np.zeros(n, dtype=np.float32)
        values  = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones   = np.array(self.dones,   dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_val         = values[t + 1] if t < n - 1 else last_value
            next_non_terminal = 1.0 - dones[t]
            delta  = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae    = delta + gamma * gae_lambda * next_non_terminal * gae
            adv[t] = gae

        returns = adv + values
        return adv, returns

    # --------------------------------------------------------- to tensors
    def to_tensors(self, gamma=0.99, gae_lambda=0.95):
        adv, ret = self.compute_gae(gamma=gamma, gae_lambda=gae_lambda)

        return dict(
            obs       = torch.FloatTensor(np.array(self.obs)),
            actions   = torch.LongTensor (np.array(self.actions)),
            log_probs = torch.FloatTensor(np.array(self.log_probs)),
            masks     = torch.BoolTensor (np.array(self.valid_masks)),
            adv       = torch.FloatTensor(adv),
            ret       = torch.FloatTensor(ret),
        )


# ══════════════════════════════════════════════════════════════════════════════
class PPOTrainer:
    """
    Manages rollout collection and PPO parameter updates.

    Hyperparameters
    ---------------
    clip_eps   : PPO clip range           (default 0.2)
    vf_coef    : value-loss coefficient   (default 0.5)
    ent_coef   : entropy bonus coefficient (default 0.01)
    gamma      : discount factor          (default 0.99)
    gae_lambda : GAE lambda               (default 0.95)
    n_epochs   : PPO update epochs        (default 4)
    batch_size : mini-batch size          (default 256)
    """

    def __init__(
        self,
        network,
        lr:         float = 3e-4,
        clip_eps:   float = 0.2,
        vf_coef:    float = 0.5,
        ent_coef:   float = 0.01,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs:   int   = 4,
        batch_size: int   = 256,
        max_grad_norm: float = 0.5,
    ):
        self.network       = network
        self.optimizer     = torch.optim.Adam(network.parameters(), lr=lr, eps=1e-5)
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.max_grad_norm = max_grad_norm

    # ──────────────────────────────────────────────── rollout collection
    def collect_rollouts(self, env, n_episodes: int):
        """
        Run `n_episodes` full episodes and fill a RolloutBuffer.
        Returns (buffer, stats_dict).
        """
        buffer = RolloutBuffer()
        wins, total_guesses = 0, 0

        for _ in range(n_episodes):
            obs, mask = env.reset()
            done = False

            while not done:
                action, log_prob, value = self.network.get_action(obs, mask)
                next_obs, reward, done, info = env.step(action)
                next_mask = env.valid_mask()

                buffer.add(obs, action, reward, done, log_prob, value, mask)

                obs  = next_obs
                mask = next_mask

            if info["won"]:
                wins += 1
            total_guesses += env.step_num

        stats = {
            "win_rate":    wins / n_episodes,
            "avg_guesses": total_guesses / n_episodes,
        }
        return buffer, stats

    # ──────────────────────────────────────────────── PPO update
    def update(self, buffer: RolloutBuffer):
        """
        Perform `n_epochs` passes of PPO over the buffer.
        Returns dict of average losses.
        """
        data = buffer.to_tensors(gamma=self.gamma, gae_lambda=self.gae_lambda)

        obs       = data["obs"]
        actions   = data["actions"]
        old_lps   = data["log_probs"]
        masks     = data["masks"]
        adv       = data["adv"]
        ret       = data["ret"]

        # Normalise advantages (mean=0, std=1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        total_pg, total_vf, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]

                new_lps, values, entropy = self.network.evaluate_actions(
                    obs[idx], actions[idx], masks[idx]
                )

                # ── policy loss ──────────────────────────────────────────
                ratio = torch.exp(new_lps - old_lps[idx])
                a     = adv[idx]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * a
                pg_loss = -torch.min(surr1, surr2).mean()

                # ── value loss ───────────────────────────────────────────
                vf_loss = F.mse_loss(values, ret[idx])

                # ── entropy bonus ────────────────────────────────────────
                ent_loss = -entropy.mean()

                # ── combined loss ────────────────────────────────────────
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pg  += pg_loss.item()
                total_vf  += vf_loss.item()
                total_ent += (-ent_loss).item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "pg_loss":  total_pg  / denom,
            "vf_loss":  total_vf  / denom,
            "entropy":  total_ent / denom,
        }