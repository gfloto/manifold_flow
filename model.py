import sys 
import torch 
import torch.nn as nn 

# simple fully connected model
# maps R^d -> R^d
class Model(nn.Module):
    def __init__(self, args, d=3):
        super().__init__()

        self.layers = args.layers
        self.hidden_dim = args.hidden_dim
        self.d = d # input and output dimension

        # define model
        self.model = nn.Sequential(
            nn.Linear(self.d+1, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()) for _ in range(self.layers - 2)],
            nn.Linear(self.hidden_dim, self.d)
        )

    def forward(self, x):
        return self.model(x)

