import sys 
import torch 

# noise at t=0 -> data at x1
# score matching to project data onto sub-manifold
class Process:
    def __init__(self, args):
        # NOTE: currently assuming plane x.sum() = 0
        self.device = args.device

    # uniform sampling init. move to importance sampling later
    def t(self):
        t = torch.rand(1).to(self.device)
        return t

    # process defined by x dt = f()dt + g()dW 
    # g() is sig * identity matrix for now
    def xt(self, x0, t):
        if t.item() < 0.5:
            # project onto manifold
            xt, mu = self.xt_1(x0, t)
        elif t.item() >= 0.5:
            xt, mu = self.xt_2(x0, t)
        return xt, mu

    # first part of diffusion, move towards sub-manifold
    def xt_1(self, x0, t):
        # get drift term (constant in time towards sub-manifold)
        f = -2*x0
        f[:, [0,1]] = 0 # only move 1 dim

        # get diffusion term (linear in time)
        g = t * torch.ones(3)
        g[2] = 0 

        # brownian noise
        dB = torch.randn_like(x0)

        # return xt 
        mu = x0 + f*t
        xt = mu #+ g*dB
        return xt, mu

    # second part of diffusion, move along sub-manifold
    def xt_2(self, x0, t):
        f = -2*x0
        f[:, 2] = 0 # move first 2 dimensions now

        # get diffusion term (linear in time)
        g = t * torch.ones(3)
        g[2] = 0

        # brownian noise
        dB = torch.randn_like(x0)

        # return xt
        mu = x0 + f*(t-0.5) 
        xt = mu #+ g*dB
        xt[:,2] = 0
        return xt, mu

    # score := grad log pdf, pdf = gaussian 
    def score(self, xt, mu):
        # NOTE: according to ddpm, we ignore variance in the score
        score = (xt - mu).pow(2)
        return score

import os
from tqdm import tqdm

from args import get_args
from dataloader import Loader
from plot import save_vis, make_gif

# visualize the forward process
if __name__ == '__main__':
    T = 50
    os.makedirs('imgs', exist_ok=True)

    # get loader and process
    args = get_args()
    args.batch_size = 2048
    loader = Loader(args)
    process = Process(args)
    
    # get initial sample
    x0 = next(iter(loader))

    # simulate forward process
    t = torch.linspace(0, 1, T).to(args.device)
    for i in tqdm(range(T)):
        # get noised data
        xt, _ = process.xt(x0, t[i])

        # save image
        save_vis(xt, f'imgs/{i}.png', i)

    # save video
    extra = 25
    for i in range(extra):
        save_vis(xt, f'imgs/{T+i}.png', T+i)
    make_gif('imgs', f'results/forward_process.gif', T+extra)

