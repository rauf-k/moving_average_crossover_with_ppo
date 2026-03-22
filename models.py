
import torch
import torch.nn as nn

import constants as CONST


class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 12, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(12, 24, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(24, 24, kernel_size=3),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            nn.Conv1d(24, 24, kernel_size=3),
            nn.ReLU(),
            # nn.MaxPool1d(2),

            # nn.AdaptiveAvgPool1d(1),  # Squashes time dimension to 1
            nn.Flatten()
        )

        # Actor Head (Continuous actions via Normal distribution)
        self.actor = nn.Sequential(
            nn.Linear(408, 102),
            nn.ReLU(),
            nn.Linear(102, 16),
            nn.ReLU(),
            nn.Linear(16, CONST.ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.zeros(CONST.ACTION_DIM))  # Trainable exploration

        # Critic Head (Predicts Value)
        self.critic = nn.Sequential(
            nn.Linear(408, 102),
            nn.ReLU(),
            nn.Linear(102, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        """
        temp_x = x
        for i, layer in enumerate(self.feature_extractor):
            temp_x = layer(temp_x)
            print(f"Layer {i} [{layer.__class__.__name__}]: {temp_x.shape}")
        exit()
        """
        mu = self.actor(features)
        value = self.critic(features)
        return mu, value

