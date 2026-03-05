"""
network.py — Wordle actor-critic network.

Architecture:
    Shared trunk : Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU
    Policy head  : Linear(256, vocab_size)   — logits (masked before softmax)
    Value head   : Linear(256, 1)            — state value estimate

Action masking is applied at inference and during PPO updates so the agent
can never select a word that's already been eliminated by feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WordleNetwork(nn.Module):

    def __init__(self, obs_dim: int, vocab_size: int, hidden_dim: int = 256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head  = nn.Linear(hidden_dim, 1)

        # Orthogonal init (empirically good for PPO)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ---------------------------------------------------------------- forward
    def forward(self, obs: torch.Tensor):
        """
        obs : (B, obs_dim)  or  (obs_dim,)
        Returns (logits, values) — values is squeezed to (B,) or scalar.
        """
        h      = self.trunk(obs)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values

    # ---------------------------------------------------- acting (no grad)
    @torch.no_grad()
    def get_action(
        self,
        obs: np.ndarray,
        valid_mask: np.ndarray,
        deterministic: bool = False,
    ):
        """
        Single-step action selection.

        Parameters
        ----------
        obs          : numpy array (obs_dim,)
        valid_mask   : boolean numpy array (vocab_size,)
        deterministic: if True, pick argmax (used at eval/play time)

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        """
        obs_t    = torch.FloatTensor(obs).unsqueeze(0)   # (1, obs_dim)
        mask_t   = torch.BoolTensor(valid_mask)           # (vocab_size,)

        logits, value = self(obs_t)
        logits = logits.squeeze(0)                        # (vocab_size,)

        # Mask out invalid actions
        logits = logits.masked_fill(~mask_t, float("-inf"))

        if deterministic:
            action = int(logits.argmax().item())
        else:
            dist   = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())

        log_prob = float(F.log_softmax(logits, dim=-1)[action].item())
        return action, log_prob, float(value.item())

    # -------------------------------------------- PPO evaluation (with grad)
    def evaluate_actions(
        self,
        obs_batch:    torch.Tensor,   # (B, obs_dim)
        action_batch: torch.Tensor,   # (B,)  long
        mask_batch:   torch.Tensor,   # (B, vocab_size)  bool
    ):
        """
        Called during the PPO update to recompute log-probs, values, entropy.
        """
        logits, values = self(obs_batch)
        logits = logits.masked_fill(~mask_batch, float("-inf"))

        dist     = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        entropy   = dist.entropy()

        return log_probs, values, entropy