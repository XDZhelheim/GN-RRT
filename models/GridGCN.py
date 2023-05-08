import torch.nn as nn
import torch
from torchinfo import summary


class GridGCN(nn.Module):
    def __init__(
        self, num_grids_width=20, num_grids_height=20, input_dim=2, output_dim=1
    ):
        super().__init__()

        self.num_grids_width = num_grids_width
        self.num_grids_height = num_grids_height
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, num_grids_width, num_grids_height, 2)
        batch_size = x.shape[0]

        out = self.fc(x)
        return out


if __name__ == "__main__":
    model = GridGCN()
    summary(model, [64, 20, 20, 2])
