import torch as th
from torch import nn
from torch.nn import functional as tf

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class BallPredictor(nn.Module):
    def __init__(self, input_dim, output_size):
        super(BallPredictor, self).__init__()
        self.input_dim = input_dim

        self.model = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, input_dim),
            'fc2': nn.Linear(input_dim, input_dim),
            'fc3': nn.Linear(input_dim, output_size),
        })
        self.to(device)

    def forward(self, x):
        x = self.model['fc1'](x)
        x = self.model['fc2'](x)
        x = self.model['fc3'](x)

        return x

class AllyPredictor(nn.Module):
    def __init__(self, input_dim, output_size):
        super(AllyPredictor, self).__init__()
        self.input_dim = input_dim

        self.model = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, input_dim),
            'fc2': nn.Linear(input_dim, input_dim),
            'fc3': nn.Linear(input_dim, output_size),
        })
        self.to(device)

    def forward(self, x):
        x = self.model['fc1'](x)
        x = self.model['fc2'](x)
        x = self.model['fc3'](x)
        return x

class EnemyPredictor(nn.Module):
    def __init__(self, input_dim, output_size):
        super(EnemyPredictor, self).__init__()
        self.input_dim = input_dim

        self.model = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, input_dim),
            'fc2': nn.Linear(input_dim, input_dim),
            'fc3': nn.Linear(input_dim, output_size),
        })
        self.to(device)

    def forward(self, x):
        x = self.model['fc1'](x)
        x = self.model['fc2'](x)
        x = self.model['fc3'](x)

        return x

class RewardPredictor(nn.Module):
    def __init__(self, input_dim, output_size):
        super(RewardPredictor, self).__init__()
        self.input_dim = input_dim

        self.model = nn.ModuleDict({
            'fc1': nn.Linear(input_dim, input_dim),
            'fc3': nn.Linear(input_dim, output_size),
        })
        self.to(device)

    def forward(self, x):
        x = self.model['fc1'](x)
        x = self.model['fc3'](x)

        return x
