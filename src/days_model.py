import torch
from torch import nn


class DaysRemainingMLP(nn.Module):
    """
    MLP regression model for predicting the number of days
    remaining until the fruit reaches fully ripe.

    Input features:
        2 fruit one-hot features
        3 stage one-hot features
        1 normalized temperature feature
        1 normalized humidity feature

    Total input features = 7

    Output:
        1 continuous value = predicted days remaining
    """

    def __init__(
        self,
        input_dim: int = 7,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError(
                "input_dim must be a positive integer."
            )

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError(
                "Input must have shape "
                "(batch_size, input_dim)."
            )

        output = self.network(x)

        return output.squeeze(1)