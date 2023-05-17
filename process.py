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
    def xt(self, x0, t, a=4):
        # noising term / diffusion        
        sig = t.sqrt() * torch.ones_like(x0)

        # step  1: project onto manifold
        if t.item() < 0.5:
            t_ = 0.25 - (t - 0.25).abs()
            sig[:, 2] = t_.sqrt()

            t = 2*t
            mu = x0 * (-a*t).exp()
            mu[:, [0,1]] = x0[:, [0,1]] # project z onto manifold

        # step 2: move torward origin on manifold
        elif t.item() >= 0.5:
            t = 2*t - 1
            mu = x0 * (-a/2*t).exp()
            mu[:, 2] = 0 # keep z on manifold

            sig[:, 2] = 0
        
        xt = mu + sig*torch.randn_like(x0)
        return xt, mu, sig

    # score := grad log pdf, pdf = gaussian 
    def score(self, xt, mu, sigma):
        score = -0.5 * ( (xt - mu) / sigma ).pow(2)
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

