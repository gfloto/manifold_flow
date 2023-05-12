import sys, os 
import torch
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
def f(x):
    x1 = x[:, 0]; x2 = x[:, 1]
    a = 0.5*torch.sin(x1) + 0.5*torch.cos(2*x2)
    b = 0.2*x1 + 0.1*x2
    c = a*b
    c -= c.mean()
    c /= c.std()

    return c[..., None]

# make dataset for training
def create_dataset(path, d=3, n=1000, k=4, vis=False):
    # let k be from k different gaussians
    x = None
    for i in range(k):
        sig = torch.rand(1)
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

# get dataset (make if one doesn't exist)
def get_dataset(path, d=3, n=10000):
    data_path = os.path.join(path, 'dummy.pt')
    if not os.path.exists(path):
        create_dataset(path, d, n)
    return torch.load(path)

# make dummy data for testing idea out
if __name__ == '__main__':
    d = 3; n = 1000; k=4

    # let k be from k different gaussians
    x = None
    for i in range(k):
        sig = torch.randn(1).abs() / 4
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
    scatter_3d(x)


