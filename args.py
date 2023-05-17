import os
import torch
import argparse

# helper to convert str to bool for argparse
def str2bool(x):
    if x in ['True', 'true']:
        return True
    elif x in ['False', 'false']:
        return False
    else:
        raise ValueError('x must be True or False')

def get_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--exp', type=str, default='dev', help='experiment name')    
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--data_path', type=str, default='data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='basic_dataset', help='dataset name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--remake_data', type=str, default='False', help='remake dataset')

    # neural network and optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--layers', type=int, default=8, help='number of layers in neural network')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of neural network')

    args = parser.parse_args()

    # get bools
    args.remake_data = str2bool(args.remake_data)

    # set folder for experiments
    args.exp = os.path.join('results', args.exp)
    os.makedirs('results', exist_ok=True)
    os.makedirs(args.exp, exist_ok=True)

    # asserts to ensure valid arguments
    assert args.device in ['cuda', 'cpu'], 'invalid device'
    assert args.hidden_dim >= 3, 'hidden dimension must be at least 3'
    assert args.layers >= 3, 'must have at least 3 layers'

    return args