import sys, os 
import torch 
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from args import get_args
from utils import save_path
from dataloader import Loader
from process import Process
from model import Model
from train import train

if __name__ == '__main__':
    args = get_args()

    # get dataset, process et. al
    loader = Loader(args)
    process = Process(args)
    model = Model(args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # option to load from checkpoint
    # TODO: have epochs saved as well...

    # train
    loss_track = []
    for epoch in range(args.epochs):
        loss = train(model, process, loader, opt, args)

        # keep track of loss for training curve
        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # save model and optimizer if best loss
        if loss == min(loss_track):
            torch.save(model.state_dict(), save_path(args, f'model.pt'))
            torch.save(opt.state_dict(), save_path(args, f'opt.pt'))

        # plot loss 
        np.save(save_path(args, 'loss.npy'), np.array(loss_track))
        plt.plot(loss_track)
        plt.yscale('log')
        plt.savefig(save_path(args, 'loss.png'))
        plt.close()