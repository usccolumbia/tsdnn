import os
import sys
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Generate training and prediction data for PU learning')
parser.add_argument('root_dir', metavar='PATH', help='path to the root directory containing the dataset CSV file')
parser.add_argument('-i', '--iter', metavar='N', type=int, default=100, help='number of iterations of to generate (default 100)')
args = parser.parse_args(sys.argv[1:])

def gen_idprop(iter):
    root_dir = args.root_dir
    assert os.path.exists(f'{root_dir}/dataset.csv'), f'Missing dataset ({root_dir}/dataset.csv)'
    if not os.path.exists(f'{root_dir}/predict'):
        os.makedirs(f'{root_dir}/predict')
    if not os.path.exists(f'{root_dir}/train'):
        os.makedirs(f'{root_dir}/train')

    dataset = pd.read_csv(f'{root_dir}/dataset.csv', names=['id', 'label'])
    p_data = dataset[dataset.label == 1].copy()
    u_data = dataset[dataset.label == 0].copy()
    p_size = p_data['id'].size
    u_size = u_data['id'].size
    predict_size = u_size - p_size
    u_sample = u_data.sample(n=p_size)
    common = u_data.merge(u_sample, on=['id'])
    u_test = u_data[~u_data.id.isin(common.id)].copy()

    u_test.reset_index(drop=True, inplace=True)
    u_test.loc[u_test.index%2==0, 'label'] = 0
    u_test.loc[u_test.index%2!=0, 'label'] = 1
    u_test.to_csv(f'{root_dir}/predict/predict_{iter}.csv', index=False, header=False)

    data = [p_data, u_sample]
    concat = pd.concat(data)
    train_data = concat.reset_index(drop=True)
    train_data.to_csv(f'{root_dir}/train/train_{iter}.csv', index=False, header=False)


def main():
    for i in range(args.iter):
        gen_idprop(i)

if __name__ == '__main__':
    main()
