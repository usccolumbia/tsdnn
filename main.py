import argparse
import operator
import random
import pandas as pd
import numpy as np
import tsdnn.train as t
import sys
import csv
import torch
import os

parser = argparse.ArgumentParser(description='Semi-Supervised Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+', help='dataset options, started with the path to root dir, then other options')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--uds', '--udsteps', default=0, type=int, metavar='N', help='number of unsupervised PU learning iterations to perform')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N', help='gpu to run on (default: 0)')
parser.add_argument('--teacher', default='', type=str, metavar='PATH', help='path to latest teacher checkpoint (default: none)')
parser.add_argument('--student', default='', type=str, metavar='PATH', help='path to latest student checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N', help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--iter', default=0, type=int, metavar='N', help='iteration number to save as (default=0)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.7, type=float, metavar='N', help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N', help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N', help='percentage of validation data to be loaded (default 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N', help='number of validation data to be loaded (default 1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N', help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N', help='number of test data to be loaded (default 1000)')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD', help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=90, type=int, metavar='N', help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=180, type=int, metavar='N', help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N', help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N', help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

root_dir = args.data_options[0]


def main():
    if args.workers > 1:
        torch.multiprocessing.set_sharing_strategy('file_system')
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    if args.uds:
        gen_datasets()

        for i in range(args.uds):
            args.uds = i
            t.main(args)
            gen_datasets()
            # regen_datasets()
            args.teacher = ''
            args.student = ''
    else:
        t.main(args)


def gen_datasets():
    assert os.path.isfile(root_dir+'/data_positive.csv'), 'missing positive dataset'
    assert os.path.isfile(root_dir+'/data_unlabeled_full.csv'), 'missing unlabeled dataset'

    positive = pd.read_csv(root_dir+'/data_positive.csv', header=None)
    positive = positive.iloc[:,0]    # ensure only labels are included
    unlabeled = pd.read_csv(root_dir+'/data_unlabeled_full.csv', header=None)
    unlabeled = unlabeled.iloc[:,0]   # ensure only labels are included

    p_size = positive.size
    u_size = unlabeled.size
    assert p_size < u_size, 'dataset is unbalanced, more positive samples than unlabeled'

    p_to_u_size = int(p_size*0.3)
    p_size -= p_to_u_size

    p_labeled = positive.sample(n=p_size).copy().reset_index(drop=True)
    common = pd.Series(list(set(positive).intersection(set(p_labeled))))
    p_unlabeled = positive[~positive.isin(common)].copy().reset_index(drop=True)    # remove duplicates

    negative = unlabeled.sample(n=p_size).copy().reset_index(drop=True)
    common = pd.Series(list(set(unlabeled).intersection(set(negative))))
    unlabeled = unlabeled[~unlabeled.isin(common)].copy().reset_index(drop=True)
    unlabeled = pd.concat([unlabeled, p_unlabeled]).reset_index(drop=True)
    unlabeled = pd.concat([unlabeled, pd.Series([x % 2 for x in range(unlabeled.size)])], axis=1)
    # print(unlabeled)

    p_labeled = pd.concat([p_labeled, pd.Series([1 for _ in range(p_labeled.size)])], axis=1)
    negative = pd.concat([negative, pd.Series([0 for _ in range(negative.size)])], axis=1)
    labeled = pd.concat([p_labeled, negative]).reset_index(drop=True)
    # print(labeled)

    labeled.to_csv(root_dir+'/data_labeled.csv', header=False, index=False)
    unlabeled.to_csv(root_dir+'/data_unlabeled.csv', header=False, index=False)


def regen_datasets():
    with open(root_dir+'/data_positive.csv', 'r') as f:
        reader = csv.reader(f)
        positive = list(reader)
        p_size = len(positive)

    with open(f'results/validation/test_results_{args.iter}.csv', 'r') as f:
        reader = csv.reader(f)
        results = dict(reader)

    with open(root_dir+f'/data_unlabeled_full.csv', 'r') as f:
        reader = csv.reader(f)
        full_unlabeled = [row[0] for row in reader]

    with open(root_dir+f'/data_unlabeled.csv', 'r') as f:
        reader = csv.reader(f)
        current_unlabeled = [row[0] for row in reader]

    unlabeled_results = dict((cid, results[cid]) for cid in full_unlabeled)
    full_unlabeled_sorted = sorted(unlabeled_results, key=unlabeled_results.get)
    current_unlabeled_sorted = [cid for cid in full_unlabeled_sorted.keys() if cid in current_unlabeled]
    current_unlabeled_results = [full_unlabeled_sorted[cid] for cid in current_unlabeled]
    current_negative = sum(pred <= 0.5 for pred in current_unlabeled_results)

    if current_negative >= p_size:
        new_negative = [(cid, 0) for cid in current_unlabeled_sorted[:p_size]]
        other_unlabeled = [cid for cid in full_unlabeled_sorted.keys() if cid not in current_unlabeled_sorted[:p_size]]
        new_unlabeled = [(cid, i%2) for i, cid in enumerate(other_unlabeled)]
    else:
        needed = p_size - current_negative
        unlabeled_negative = [(cid, 0) for cid in current_unlabeled_sorted[:current_negative]]
        labeled_negative = random.sample()
        unlabeled = [row[0] for row in reader]

    unlabeled_results = dict((cid, results[cid]) for cid in unlabeled)
    unlabeled_sorted = sorted(unlabeled_results, key=unlabeled_results.get)

    with open(root_dir+'/data_positive.csv', 'r') as f:
        reader = csv.reader(f)
        positive = list(reader)

    p_size = len(positive)
    new_negative = [(cid, 0) for cid in unlabeled_sorted[:p_size]]
    new_labeled = positive + new_negative
    new_unlabeled = [(cid, i%2) for i, cid in enumerate(full_unlabeled_sorted[p_size:])]

    with open(root_dir+'/data_labeled.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_labeled)

    with open(root_dir+'/data_unlabeled.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_unlabeled)



if __name__ == '__main__':
    main()
