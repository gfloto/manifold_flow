import sys, os 
import shutil
import torch
import numpy as np 
import imageio
import matplotlib.pyplot as plt 
plt.style.use('seaborn')

from utils import ptnp

# save data as a 3d scatter plot
def save_vis(x, path, i):
    x = ptnp(x)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    # set camera angle
    ax.view_init(0.4*i + 10, 0.75*i)

    ax.plot(x[:, 0], x[:, 1], x[:, 2], 'o', markersize=2, alpha=0.5)
    plt.savefig(path)
    plt.close()

# make gif from images, name is f'path/{}.png' from 0 to n
def make_gif(path, name, n):
    print('making gif...')
    images = []
    for i in range(n):
        images.append(imageio.imread(os.path.join(path, f'{i}.png')))
    imageio.mimsave(f'{name}', images)

    # remove images and folder
    shutil.rmtree('imgs')
