import sys, os 
import torch
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('seaborn')

from utils import ptnp

# 3d scatter plot of data
def scatter_3d(x):
    x = ptnp(x) # convert to numpy

    # plot 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:,2], 'o', markersize=3, alpha=0.5)
    plt.show()

# make some complex 3d data
def f_old(x):
    x1 = x[:, 0]; x2 = x[:, 1]
    a = 0.5*torch.sin(x1) + 0.5*torch.cos(2*x2)
    b = 0.2*x1 + 0.1*x2
    c = a*b
    c -= c.mean()
    c /= c.std()

    return c[..., None]

# a simple gauss w plane
def f(x):
    norm = -x.norm(dim=1, keepdim=True).pow(2)
    return 2*norm.exp() + 0.25*x.sum(dim=1, keepdim=True)

# make dataset for training
def create_dataset(path, d=3, n=1000, k=4, vis=False):
    # let k be from k different gaussians
    x = None
    for i in range(k):
        sig = torch.rand(1) + 0.5
        loc = torch.randn(1, d-1)
        x_ = torch.randn(n, d-1) * sig + loc 

        if x is None:
            x = x_
        else:
            x = torch.cat([x, x_], dim=0)

    # lift to higher dimension
    y = f(x)

    # treat data as 3d
    x = torch.cat([x, y], dim=1)
    if vis:
        scatter_3d(x)

    # save data
    torch.save(x, path)

# make swirl dataset for testing more difficult manifolds
def make_swirl(n, d):
    radial_init = 0.5
    radial_v = 0.25 # radial distance per radian
    rot_rad = 3*np.pi # total radians to rotate

    # make swirl on xy plane, then lift into y
    len_ = torch.rand(n)
    angle = rot_rad * len_
    dist = radial_v * angle + radial_init

    # make xy plane
    x = dist[..., None] * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)

    # make y
    y = torch.randn(n, d-2)
    x = torch.cat([x, y], dim=1)
    return x

# get dataset (make if one doesn't exist)
def get_dataset(path, d=3, n=10000):
    data_path = os.path.join(path, 'dummy.pt')
    if not os.path.exists(path):
        create_dataset(path, d, n)
    return torch.load(path)

# make dummy data for testing idea out
if __name__ == '__main__':
    n = 5000
    d = 3
    dataset = 'swirl'

    if dataset == 'simple':
        # let k be from k different gaussians
        x = None
        for i in range(k):
            sig = torch.randn(1).abs() + 0.5
            loc = torch.randn(1, d-1)
            x_ = torch.randn(n, d-1) * sig + loc 

            if x is None:
                x = x_
            else:
                x = torch.cat([x, x_], dim=0)

            # lift to higher dimension
            y = f(x)

            # treat data as 3d
            x = torch.cat([x, y], dim=1)
    
    elif dataset == 'swirl':
        x = make_swirl(n, d)

    scatter_3d(x)

