# Positive and Unknown Material Synthesizability Prediction

This is an unofficial implementation of the [Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning](https://pubs.acs.org/doi/10.1021/jacs.0c07384?ref=pdf) positive and unknown learning method built on the [Crystal Graph Convolutional Neural Network](https://github.com/txie-93/cgcnn) (CGCNN) to take an arbitary crystal structure to predict material synthesizability. 

The package provides two major functions:

- Train a semi-supervised CGCNN model with a customized dataset.
- Predict material synthesizability of new crystals with a pre-trained CGCNN model.

The following paper describes the details of the CGCNN architecture:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

The following papers describe the details of the semi-supervised (PU) learning method:

[Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning](https://pubs.acs.org/doi/10.1021/jacs.0c07384?ref=pdf)  
[A bagging SVM to learn from positive and unlabeled examples](https://arxiv.org/abs/1010.0772)

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a CGCNN model](#train-a-cgcnn-model)
- [Results](#results)
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
conda create -n cgcnn python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
```

*Note: this code is tested for PyTorch v1.0.0+ and is not compatible with versions below v0.4.0 due to some breaking changes.

This creates a conda environment for running CGCNN. Before using CGCNN, activate the environment by:

```bash
source activate cgcnn
```

Then, in the directory `pu-cgcnn`, you can test if all the prerequisites are installed properly by running:

```bash
python main.py -h
```

This should display the help message. If you find no error messages, it means that the prerequisites are installed properly.

After you finished using the CGCNN, exit the environment by:

```bash
source deactivate
```

## Usage

### Define a customized dataset 

To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `data/root_dir` with the following files: 

1. `dataset.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property (1 for positive and 0 for unknown).

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/root_dir/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── dataset.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

To replicate our results, use the dataset found in `data/root_dir/dataset.csv`. 

### Train a CGCNN model

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.
- Generate the necessary training files using `python gen_idprops.py -i N root_dir` where `N` is the number of iterations to generate and `root_dir` is the root directory conntaining the `dataset.csv` file.

Then, in directory `pu-cgcnn`, you can train a CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. You can set the iteration to use by the flag `--iter`.

After training, you will get four files in `pu-cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `results/validation/test_results_{iteration}.csv`: stores the `ID` and predicted classification score for each positive crystal in test set.
- `results/predictions/predictions_{iteration}.csv`: stores the `ID` and predicted classification score for each unknown crystal in prediction set.

## Results

We have replicated the results found in the [Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning](https://pubs.acs.org/doi/10.1021/jacs.0c07384?ref=pdf) paper.

|  | Paper w/ 2015 Dataset | Paper w/ Inital Structures | Paper/ Relaxed Structures | Paper | This Code |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 86.2 | 87.36 | 87.41 | 87.4 | 87.33 |

## Authors

This software was primarily written by [Daniel Gleaves](http://mleg.cse.sc.edu/people.html) who was advised by [Prof. Jianjun Hu](https://cse.sc.edu/~jianjunh/).  
This was built upon the CGCNN written by [Tian Xie](http://txie.me/) who was advised by [Prof. Jeffrey Grossman](https://dmse.mit.edu/faculty/profile/grossman).

## License

CGCNN is released under the MIT License.



