import sys, os
import argparse
import torch
import numpy as np 

from make_data import create_dataset

# efficient data loading
class Loader:
    def __init__(self, args, shuffle=True):
        self.device = args.device
        self.shuffle = shuffle
        
        # make dataset if doesn't exist
        data_path = os.path.join(args.data_path, f'{args.dataset}.pt')
        if not os.path.exists(data_path) or args.remake_data:
            create_dataset(data_path, vis=True)

        # load data 
        print('loading data from', data_path)
        self.x = torch.load(data_path)
        
        self.batch_size = args.batch_size
        self.n_batches = int(np.ceil(self.x.shape[0] / self.batch_size))
        self.count = 0

        # index dat via self.ind
        if self.shuffle:
            self.ind = np.random.permutation(np.arange(self.x.shape[0]))
        else:
            self.ind = np.arange(self.x.shape[0])

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.n_batches:
            x = self.get_batch(self.count)
            self.count += 1
            return x.to(self.device)
        else:
            self.count = 0
            if self.shuffle:
                self.shuffle_ind()
            raise StopIteration

    def get_batch(self, i):
        if i != self.n_batches - 1:
            ind = self.ind[i * self.batch_size : (i + 1) * self.batch_size]
        else:
            ind = self.ind[i * self.batch_size : ]

        x = self.x[ind]
        return 2*x

    # call this at end of get_batch
    def shuffle_ind(self):
        self.ind = np.random.permutation(self.ind)
