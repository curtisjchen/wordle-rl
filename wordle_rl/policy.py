import torch
import torch.nn as nn


class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,action_dim)
        )

    def forward(self,x):

        return self.net(x)