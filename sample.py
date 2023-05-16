import sys, os
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

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
    def init_sample(self):
        # place on plane z=0
        x = torch.randn(self.batch_size, self.data_shape).to(self.device)
        x[:,2] = 0
        return x

    # f(x, t); drift term
    def f(x, t):
        f = -2*x
        if t.item() < 0.5:
            f / t
            f[:, [0,1]] = 0
        elif t.item() >= 0.5:
            f / (t-0.5)
            f[:, 2] = 0

        return f

    # g(x): diffusion term
    def g(x):
        g = torch.ones_like(x).to(self.device)
        g[:, 2] = 0
        
        return g

    # backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dB
    def update(self, model, x, t, dt, g_scale=1):
        # get f, g, g^2 score and dB
        f = self.process.sde_f(x)
        g = self.process.sde_g(x)     

        # set t to tensor, then get score
        t = torch.tensor([t]).float().to(self.device)
        g2_score = model(x, t)

        # check f is not nan
        assert torch.isnan(f).sum() == 0, f'f is nan: {f}'
        dB = (np.sqrt(dt) * torch.randn_like(g2_score)).to(self.device) 

        # solve sde: https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method 
        gdB = torch.einsum('b i j ..., b j ... -> b i ...', g, dB)
        return (-f + g2_score)*dt + g_scale*gdB

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png'):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # initial time is t
        t = torch.tensor([1.]).to(self.device)
        dt = t / T

        # noise schedule
        g_scale = np.linspace(0,1,T)[::-1]
        g_scale = 1.75*np.power(g_scale, 1.5)

        # sample loop
        d = 50
        for i in tqdm(range(T)):
            # update x
            change = self.update(model, x, t[i], dt[i], g_scale[i])
            x = x + change

            # save sample
            if i % d == 0:
                save_vis(x.clone(), f'imgs/{int(i/d)}.png',i/d)

        # discretize
        for i in range(int(T/d), int(T/d)+10):
            save_vis(x, f'imgs/{i}.png', i/d)

        # save gif
        make_gif('imgs', save_path, int(T/d)+10)

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
    batch_size = 128
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
