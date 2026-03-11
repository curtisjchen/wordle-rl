import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for PPO stability."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.activation = nn.ELU()

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.fc(x)
        x = self.activation(x)
        return x + residual

class WordleNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=1024):
        super().__init__()
        
        # 1. Independent Actor (Policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ELU(),
            ResidualBlock(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01) # Small std for exploration
        )
        
        # 2. Independent Critic (Value)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ELU(),
            ResidualBlock(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x, mask=None):
        logits = self.actor(x)
        value = self.critic(x)
        
        if mask is not None:
            # Safer negative constant to avoid NaN in log_softmax/mixed precision
            logits = torch.where(mask, logits, torch.tensor(-1e8).to(logits.device))
            
        return logits, value

    def get_action(self, obs, mask=None, deterministic=False):
        """
        Helper for collection and evaluation.
        Handles masking and distribution sampling in one call.
        """
        logits, value = self.forward(obs, mask)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            return action, None, value
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value