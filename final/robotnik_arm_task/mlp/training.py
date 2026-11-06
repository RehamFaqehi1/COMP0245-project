"""Shared MLP architecture for training and inference."""

from torch import nn


class MLP(nn.Module):
    """Shared 128-128-64 ReLU regressor with eleven command outputs.

    Args:
        input_size: Feature width: 33, 35, 37 or 70 for the supported schemas.

    The output order is four wheel commands followed by seven arm torques;
    input scaling and output inverse scaling are handled by the caller.
    """
    def __init__(self, input_size=33):
        super().__init__()
        if input_size not in (33,35,37,70):
            raise ValueError('Supported MLP input widths are 33, 35, 37 and 70')
        self.network = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 11))

    def forward(self, x):
        return self.network(x)
