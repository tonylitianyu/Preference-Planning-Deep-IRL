import numpy as np
import torch
import torch.nn.functional as F


def maxentirl_loss(learner, expert, reward_func, device):
    learner_torch = torch.FloatTensor(learner).to(device)
    expert_torch = torch.FloatTensor(expert).to(device)

    learner_r = reward_func.r(learner_torch).view(-1)
    expert_r = reward_func.r(expert_torch).view(-1)


    return 1000 * (learner_r.mean() - expert_r.mean())