import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleReward(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(SimpleReward, self).__init__()
        self.l1=nn.Linear(n_input, n_hidden)
        self.l2=nn.Linear(n_hidden, n_hidden)
        self.l3=nn.Linear(n_hidden, 1)
        

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def r(self, batch):
        return self.forward(batch)

    def normalize_state(self, obs_high, obs_low, state_sample):
        space = obs_high - obs_low
        norm_sample = state_sample/space
        return norm_sample

