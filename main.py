import sys, os 
import json
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

# sample arguments as json
def save_args(args):
    # save args as json
    args_dict = vars(args)
    os.makedirs(args.exp, exist_ok=True)
    with open(save_path(args, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

if __name__ == '__main__':
    # load and save args
    args = get_args()
    save_args(args)

    # get dataset, process et. al
    loader = Loader(args)
    process = Process(args)
    model = Model(args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    loss_track = []
    save_flag = False
    for epoch in range(args.epochs):
        loss = train(model, process, loader, opt, args)

        # keep track of loss for training curve
        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # save model and optimizer if best loss
        if loss == min(loss_track):
            save_flag = True

        # plot loss and save model
        if epoch % 100 == 0:
            np.save(save_path(args, 'loss.npy'), np.array(loss_track))
            plt.plot(loss_track)
            plt.yscale('log')
            plt.savefig(save_path(args, 'loss.png'))
            plt.close()

            if save_flag:
                save_flag = False
                print('saving model...')
                torch.save(model.state_dict(), save_path(args, f'model.pt'))
                torch.save(opt.state_dict(), save_path(args, f'opt.pt'))
