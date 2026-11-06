"""Neural-network architecture."""

from torch import nn
from mlp_config import HIDDEN_LAYERS


class MLP(nn.Module):
    # Map robot state and target position to wheel commands and arm torques.

    def __init__(self, input_size=20, output_size=11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[2], output_size),
        )

    def forward(self, x):
        return self.network(x)
