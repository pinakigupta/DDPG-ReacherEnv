import numpy as np
from importlib import reload 
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.ModuleList([])
        fc_unit_input_size = state_size
        for fc_unit_size in fc_units:
            self.fc.append(nn.Linear(fc_unit_input_size, fc_unit_size))
            fc_unit_input_size = fc_unit_size
        self.fc.append(nn.Linear(fc_unit_input_size, action_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.fc)):
            self.fc[i].weight.data.uniform_(*hidden_init(self.fc[i]))
        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for fc_net in self.fc:
            y = fc_net(x)
            x = F.relu(y)
        return F.tanh(y)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.ModuleList([])
        fc_unit_input_size = state_size
        for i in range(len(fc_units)):
            fc_unit_size = fc_units[i]
            self.fc.append(nn.Linear(fc_unit_input_size, fc_unit_size))
            fc_unit_input_size = fc_unit_size
            if (i == 1):
                fc_unit_input_size += action_size
        self.fc.append(nn.Linear(fc_unit_input_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.fc)):
            self.fc[i].weight.data.uniform_(*hidden_init(self.fc[i]))
        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = state
        for i in range(len(self.fc)):
            fc_net = self.fc[i]
            y = fc_net(x)
            x = F.relu(y)
            if (i == 1):
                x = torch.cat((x, action), dim=1)
        return y
