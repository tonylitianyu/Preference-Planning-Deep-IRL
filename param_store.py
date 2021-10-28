import torch
import os


def save_checkpoint(checkpoint, name):
    if not os.path.exists('trained_models/'):
        os.makedirs('trained_models/')

    torch.save(checkpoint, "trained_models/"+name)

def load_checkpoint(name):
    checkpoint = torch.load("trained_models/"+name)
    return checkpoint