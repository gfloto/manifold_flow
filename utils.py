import sys, os
import torch
import numpy as np

# convenience
def ptnp(x):
    return x.detach().cpu().numpy()

# append path to experiment folder
def save_path(args, path):
    return os.path.join(args.exp, path)
