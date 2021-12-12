# Deep Semi-Supervised Teacher-Student Material Synthesizability Prediction

Citation: 

Semi-supervised teacher-student deep neural network for materials discovery” by Daniel Gleaves, Edirisuriya M. Dilanga Siriwardane,Yong Zhao, and JianjunHu.

Machine learning and evolution laboratory

Department of Computer Science and Engineering

University of South Carolina

<hr>

This software package implements the Meta Pseudo Labels (MPL) semi-supervised learning method with Crystal Graph Convolutional Neural Networks (CGCNN) with that takes an arbitary crystal structure to predict material synthesizability and whether it has positive or negative formation energy

The package provides two major functions:

- Train a semi-supervised TSDNN classification model with a customized dataset.
- Predict material synthesizability and formation energy of new crystals with a pre-trained TSDNN model.

The following paper describes the details of the CGCNN architecture, a graph neural network model for materials property prediction: [CGCNN paper](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

The following paper describes the details of the semi-supervised learning framework that we used in our model: [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a TSDNN model](#train-a-TSDNN-model)
  - [Classify materials with a pre-trained TSDNN model](#classify-materials-with-a-pre-trained-TSDNN-model)
- [Data](#data)
- [Authors](#authors)
- [License](#license)


##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `cgcnn` and install all prerequisites:

```bash
conda upgrade conda
conda create -n tsdnn python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
```

*Note: this code is tested for PyTorch v1.0.0+ and is not compatible with versions below v0.4.0 due to some breaking changes.

This creates a conda environment for running TSDNN. Before using TSDNN, activate the environment by:

```bash
conda activate tsdnn
```

## Usage

### Define a customized dataset 

To input crystal structures to TSDNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target label for each crystal (not needed for predicting, but you need to put some random numbers in `data_test.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `data_labeled.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the known value of the target label. 

1. `data_unlabeled.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column can be filled with alternating 1 and 0 (the second column is still needed).

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

(4.) `data_predict`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column can be filled with alternating 1 and 0 (the second column is still needed). 
This is the file that will be used if you want to classify materials with `predict.py`. 

The structure of the `root_dir` should be:

```
root_dir
├── data_labeled.csv
├── data_unlabeled.csv
├── data_test.csv
├── data_positive.csv (optional- for positive and unlabeled dataset generation)
├── data_unlabeled_full.csv (optional- for positive and unlabeled dataset generation, data_unlabeled.csv will be overwritten)
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There is an example of customized a dataset in: `data/example`.

### Train a TSDNN model

Before training a new TSDNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in directory `synth-tsdnn`, you can train a TSDNN model for your customized dataset by:

```bash
python main.py root_dir
```

If you want to use the PU learning dataset generation, you can train a model using the `--uds` flag with the number of PU iterations to perform.

```bash
python main.py --uds 5 root_dir
``` 


You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/example` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 6 --val-size 2 --test-size 2 data/example
```
or alternatively
```bash
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/example
```

After training, you will get 5 files in `synth-tsdnn` directory.

- `checkpoints/teacher_best.pth.tar`: stores the TSDNN teacher model with the best validation accuracy.
- `checkpoints/student_best.pth.tar`" stores the TSDNN student model with the best validation accuracy.
- `checkpoints/t_checkpoint.pth.tar`: stores the TSDNN teacher model at the last epoch.
- `checkpoints/s_checkpoint.pth.tar`: stores the TSDNN student model at the last epoch.
- `results/validation/test_results.csv`: stores the `ID` and predicted value for each crystal in training set.

### Predict material properties with a pre-trained TSDNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a pre-trained TSDNN model (example found in checkpoints/pre-trained/pre-train.pth.tar).

Then, in directory `synth-tsdnn`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py checkpoints/pre-trained/pre-trained.pth.tar data/root_dir
```

After predicting, you will get one file in `synth-tsdnn` directory:

- `predictions.csv`: stores the `ID` and predicted value for each crystal in test set.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](data/material-data). Each dataset discussed can be found in `data/datasets/`

## Authors

This software was primarily written by [Daniel Gleaves](http://mleg.cse.sc.edu/people.html) who was advised by [Prof. Jianjun Hu](https://cse.sc.edu/~jianjunh/index.html). This software builds upon work by [Tian Xie](https://github.com/txie-93), [Hieu Pham](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels), and [Jungdae Kim](https://github.com/kekmodel).

## Acknowledgements

Research reported in this work was supported in part by NSF under grants 1940099 and 1905775. The views, perspective,and content do not necessarily represent the official views of NSF. This work was supported in part by the South Carolina Honors College Research Program. This work is partially supported by a grant from the University of South Carolina Magellan Scholar Program.

## License

TSDNN is released under the MIT License.



