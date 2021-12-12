import os
import csv
import pathlib
import model.predict as p
from operator import truediv
from shutil import copy


def init():
    global datapath, root_dir, cwd, pid, cif
    datapath = pathlib.Path(__file__).parent.absolute()/'data'
    root_dir = pathlib.Path(__file__).parent.absolute()


def avg_results():
    ids = []
    preds = []
    counter = []

    for filename in os.listdir(root_dir/'predictions'):
        if filename.endswith('.csv'):
            with open(root_dir/'predictions'/filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] not in ids:
                        ids.append(row[0])
                        preds.append(float(row[1]))
                        counter.append(1)
                    else:
                        i = ids.index(row[0])
                        preds[i] += float(row[1])
                        counter[i] += 1
            os.remove(root_dir/'predictions'/filename)

    avg_preds = list(map(truediv, preds, counter))
    return avg_preds[0]


def predict():
    init()
    p.predict([str(root_dir/'checkpoints'), str(datapath)], root_dir)
    os.remove(datapath/cif)
    return avg_results()
