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
    # f() is vector from x0 to nearest point on submanifold
    # g() is sig * identity matrix for now
    def xt(self, x0, t, g_min=0.01):
        # get drift term (constant in time towards sub-manifold)
        f = -x0
        f[:, [0,1]] = 0 # only move 1 dim

        # get diffusion term (linear in time)
        g_scale = max(g_min, t)
        g = g_scale * torch.ones(3)
        g[2] = g_min 

        # brownian noise
        dB = torch.randn_like(x0)

        # return xt 
        xt = x0 + f*t #+ g*dB
        mu = x0 + f*t
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

