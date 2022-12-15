from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def split_bagging(id_prop_folder, bagging_size, folder, gen_test=True):
    df = pd.read_csv(os.path.join(id_prop_folder, 'dataset.csv'), header=None)

    # Split positive/unlabeled data
    exp = []
    vir = []
    for i in range(len(df)):
        if df[1][i] == 1:
            exp.append(df[0][i])
        elif df[1][i] == 0:
            vir.append(df[0][i])
        else:
            raise Exception("ERROR: prop value must be 1 or 0")

    positive = pd.DataFrame()
    positive[0] = exp
    positive[1] = [1 for _ in range(len(exp))]

    unlabeled = pd.DataFrame()
    unlabeled[0] = vir
    unlabeled[1] = [0 for _ in range(len(vir))]

    # Sample positive data for validation and training
    if gen_test:
        valid_positive = positive.sample(frac=0.2, random_state=1234)
        train_positive = positive.drop(valid_positive.index)
    else:
        train_positive = positive

    os.makedirs(folder, exist_ok=True)

    # Sample negative data for training
    for i in tqdm(range(bagging_size), desc='Bagging iterations'):
        # Randomly labeling to negative
        negative = unlabeled.sample(n=len(positive[0]))
        if gen_test:
            valid_negative = negative.sample(frac=0.2, random_state=1234)
            train_negative = negative.drop(valid_negative.index)
            valid = pd.concat([valid_positive, valid_negative])

        else:
            train_negative = negative
            valid = pd.read_csv(os.path.join(
                id_prop_folder, 'data_test.csv'), header=None)

        valid.to_csv(os.path.join(folder, 'data_test_' + str(i) + '.csv'),
                     mode='w', index=False, header=False)
        train = pd.concat([train_positive, train_negative])
        train.to_csv(os.path.join(folder, 'data_labeled_' + str(i) + '.csv'),
                     mode='w', index=False, header=False)

    # Generate unlabeled data
        test_unlabel = unlabeled.drop(negative.index)
        test_unlabel.to_csv(os.path.join(
            folder, 'data_unlabeled_' + str(i)) + '.csv', mode='w', index=False, header=False)


def bootstrap_aggregating(iteration, bagging_size, prediction=False):

    predval_dict = {}

    for i in tqdm(range(bagging_size), desc='Aggregating results'):
        if prediction:
            filename = 'results/predictions/predictions_' + \
                str(iteration) + '.csv'
        else:
            filename = 'results/validation/test_results_' + \
                str(iteration) + '.csv'
        df = pd.read_csv(os.path.join(filename), header=None)
        id_list = df.iloc[:, 0].tolist()
        pred_list = df.iloc[:, 1].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predval_dict:
                predval_dict[mat_id].append(float(pred_list[idx]))
            else:
                predval_dict[mat_id] = [float(pred_list[idx])]

    print("Writing results to file...")
    with open('test_results_ensemble_' + str(iteration) + '_' + str(bagging_size) + 'models.csv', "w") as g:
        # mp-id, CLscore, # of bagging size
        g.write("id,score,bagging")

        for key, values in predval_dict.items():
            g.write('\n')
            g.write(key + ',' + str(np.mean(np.array(values))) +
                    ',' + str(len(values)))
    print("Done")


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(self, root_dir, labeled=False, max_num_nbr=12, radius=16, dmin=0, step=0.2,
                 random_seed=123, predict=False, uds=False):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        if uds:
            id_prop_file = os.path.join(self.root_dir, 'dataset.csv')
        elif labeled:
            id_prop_file = os.path.join(self.root_dir, 'data_labeled.csv')
        elif predict:
            id_prop_file = os.path.join(self.root_dir, 'data_test.csv')
        else:
            id_prop_file = os.path.join(self.root_dir, 'data_unlabeled.csv')
        assert os.path.exists(id_prop_file), f'{id_prop_file} does not exist!'

        random.seed(random_seed)
        positive_data = []
        unlabeled_data = []
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            if uds:
                for row in reader:
                    if int(row[1]) == 1:
                        positive_data.append(row)
                    elif int(row[1]) == 0:
                        unlabeled_data.append(row)
                    else:
                        raise Exception("ERROR: dataset value must be 1 or 0")
                random.shuffle(unlabeled_data)
                self.pos_len = len(positive_data)
                self.id_prop_data = positive_data + unlabeled_data
            else:
                self.id_prop_data = [row for row in reader]
                random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
