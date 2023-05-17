import sys, os
from tqdm import tqdm
import torch
import numpy as np

from utils import ptnp

def train(model, process, loader, opt, args):
    device = args.device

    loss_track = []
    for i, x0 in enumerate(loader):
        # get t, x0 xt and score
        t = process.t() # get scaled and unscaled t
        xt, mu, sig = process.xt(x0, t)

        sig = sig.clamp(min=1e-4)
        score = process.score(xt, mu, sig)

        # input as (x, t)
        t_ = t * torch.ones(xt.shape[0], 1).to(args.device)
        inp = torch.cat([xt, t_], dim=1)

        # predict score
        score_model = model(inp)

        # loss
        loss = (score_model - score).pow(2).mean()

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # save loss
        loss_track.append(loss.item())
    return np.mean(loss_track)
