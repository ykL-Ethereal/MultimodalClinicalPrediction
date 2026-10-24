from torch import nn


class StaticEncoder(nn.Module):
    def __init__(self, dimension=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, dimension), nn.ReLU(), nn.Linear(dimension, dimension), nn.ReLU())

    def forward(self, x):
        return self.network(x)

