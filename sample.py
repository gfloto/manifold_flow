import sys, os
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from process import gamma 
from model import Model
from plot import save_vis, make_gif
from utils import save_path

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

t < 0.5:
f(x,t)_{0,1} = 0
f(x,t)_{2} = -2x

t >= 0.5:
f(x,t)_{0,1} = -2x
f(x,t)_{2} = 0

g(x,t)_{0,1} = t
g(x,t)_{2} = 0
'''

class Sampler:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size

        # shapes for image vs text datasets
        self.data_shape = 3 # 3d test data for now

    # get intial sample from plane
    def init_sample(self, a=8):
        # place on plane z=0
        var = 0.5 * ( 1 - np.exp(-a) )
        x = torch.randn(self.batch_size, self.data_shape).to(self.device)
        x *= np.sqrt(var)
        x[:,2] = 0
        return x

    # f(x, t); drift term
    def f(self, x, t, a=8):
        if t.item() < 0.5:
            f = -a*x
            f[:, [0,1]] = 0

        elif t.item() >= 0.5:
            f = -a*x
            f[:, 2] = 0

        return f

    # g(x): diffusion term
    def g(self, x, t, a=8):
        g = torch.ones_like(x).to(self.device)
        if t.item() < 0.5:
            g[:, 2] = gamma(t)

        elif t.item() >= 0.5:
            g *= np.sqrt(a)
            g[:, 2] = 0
        
        return g

    # backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dB
    def update(self, model, x, t, dt):
        # get f, g, g^2 score and dB
        f = self.f(x, t)
        g = self.g(x, t)     

        # include t in input
        t_ = t * torch.ones(x.shape[0], 1).to(self.device)
        x_inp = torch.cat([x, t_], dim=1)
        score = model(x_inp)

        # brownian noise
        g_ = g.clone()
        g_[:, 2] = 0
        dB = dt.sqrt() * torch.randn_like(x).to(self.device)
        return (-f + g.pow(2)*score)*dt + 0.5*g_*dB

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png'):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # initial time is t
        t = torch.tensor([1.]).to(self.device)
        dt = t / T

        # sample loop
        d = 50
        for i in tqdm(range(T)):
            # update x
            change = self.update(model, x, t, dt)
            x = x + change
            t -= dt

            # save sample
            if i % d == 0:
                save_vis(x.clone(), f'imgs/{int(i/d)}.png', i/d)

        # save gif
        save_vis(x.clone(), f'imgs/{int(i/d)}.png', i/d, show=True)
        make_gif('imgs', save_path, int(T/d))

import json
import argparse

# to load experiment
def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, help='experiment name')
    args = parser.parse_args()

    assert args.exp is not None, 'Must specify experiment name'
    return args

if __name__ == '__main__':
    batch_size = 5000
    sample_args = get_sample_args()
    path = os.path.join('results', sample_args.exp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    model = Model(args).to(args.device)
    model_path = os.path.join('results', sample_args.exp, f'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    print(f'Loaded model from {model_path}')

    # sample from model
    sampler = Sampler(batch_size, device)
    sampler(model, T=2000, save_path=save_path(args, 'sample.gif'))
