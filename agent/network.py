import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WordleNetwork(nn.Module):
    """
    Fast MLP Actor-Critic for Wordle.

    Replaces the previous transformer architecture with a 3-layer MLP.
    Reasons:
      - Input is now a 183-dim float knowledge-state (not raw tile indices),
        so learned attention over tile positions adds no value.
      - MLP forward passes are ~5-10x faster on CPU for this input size.
      - The knowledge-state representation already encodes the relational
        information the transformer was trying to learn (e.g. "A is absent").

    Architecture:
        Linear(obs_dim → hidden) → ReLU
        Linear(hidden → hidden)  → ReLU
        Linear(hidden → hidden//2) → ReLU
              ↓                 ↓
        policy_head          value_head
        Linear(→vocab_size)  Linear(→1)
    """

    def __init__(self, obs_dim: int, vocab_size: int, hidden_dim: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        mid = hidden_dim // 2

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, mid), nn.ReLU(),
        )

        self.policy_head = nn.Linear(mid, vocab_size)
        self.value_head  = nn.Linear(mid, 1)

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight initialisation (orthogonal — standard for PPO)             #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            obs:  (B, 183) float32 knowledge-state tensor.
            mask: (B, vocab_size) bool — True for valid (guessable) actions.

        Returns:
            logits: (B, vocab_size)  — raw policy scores (invalid actions = -1e8)
            values: (B,)             — state-value estimates
        """
        x      = self.shared(obs.float())
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)

        return logits, values

    # ------------------------------------------------------------------ #
    #  Inference helper (no grad, handles numpy / batching)              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_action(self, obs, mask, deterministic: bool = False):
        """
        Convenience wrapper for rollout collection and evaluation.
        Accepts numpy arrays or tensors, batched or single observations.

        Returns numpy arrays: (actions, log_probs, values)
        """
        device = next(self.parameters()).device

        # -- obs --
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.float().to(device)

        # -- mask --
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        else:
            mask = mask.bool().to(device)

        # Handle unbatched single observations
        if obs.dim() == 1:
            obs  = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)

        logits, values = self(obs, mask)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()

        log_probs = F.log_softmax(logits, dim=-1)
        chosen_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        return actions.cpu().numpy(), chosen_lp.cpu().numpy(), values.cpu().numpy()