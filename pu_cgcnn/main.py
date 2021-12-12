import argparse
import pandas as pd
import numpy as np
import cgcnn.train as t
import cgcnn.predict as p
import sys
import csv
import torch
from os import path

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                    help='gpu to run on (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--iter', '-i', default=0, type=int, metavar='N',
                    help='iteration number to save as (default=0)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.0, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.2)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=90, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=180, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

root_dir = args.data_options[0]

# Train
# python main.py --iter 100 --train-ratio 0.8 --val-ratio 0.0 --test-ratio 0.2 data/root_dir

# Debug
# python main.py --iter 3 --epochs 50 -p 1 --atom-fea-len 90 --h-fea-len 180 -b 50 --lr 0.001 --train-ratio 0.008 --val-ratio 0.0 --test-ratio 0.002 data/root_dir

def main():
    i = args.iter
    if args.workers > 1:
        torch.multiprocessing.set_sharing_strategy('file_system')
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    print(f'***TRAINING [{i}]***')
    if path.exists(f'checkpoints/checkpoint_{i}.pth.tar'):
        args.resume = f'checkpoints/checkpoint_{iteration}.pth.tar'
        test_cif_ids, test_labels, test_preds = t.main(args, iteration)
    else:
        test_cif_ids, test_labels, test_preds = t.main(args, iteration)
    while(0 in test_labels):
        ix = test_labels.index(0)
        del test_cif_ids[ix]
        del test_labels[ix]
        del test_preds[ix]

    with open(f'results/validation/test_results_{i}.csv', 'w') as f:
        writer = csv.writer(f)
        for cif_id, pred in zip(test_ids, test_values):
            writer.writerow((cif_id, pred))

    print(f'***PREDICTING [{i}]***')
    pred_ids, pred_values = p.predict([f'checkpoints/checkpoint_{iteration}.pth.tar', root_dir], iteration)

    with open(f'results/predictions/predictions_{i}.csv', 'w') as f:
        writer = csv.writer(f)
        for cif_id, pred in zip(pred_ids, pred_values):
            writer.writerow((cif_id, pred))


if __name__ == '__main__':
        main()
