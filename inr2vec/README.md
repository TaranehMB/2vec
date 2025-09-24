# Deep Learning on INRs of Shapes

Official code for the paper [Deep Learning on Implicit Neural Representations of Shapes](https://arxiv.org/abs/2302.05438), published 
at ICLR 2023 and its extended paper, code related to **nf2vec** framework, which is detailed in the paper [Deep Learning on Object-centric 3D Neural Fields](https://arxiv.org/abs/2312.13277). In this repository, you would find both codes for the framework necessary to process 3D shapes and the framework to process NeRFs. 

The papers corresponding to this repository and their seperate frameworks can be found in link below:

[[Web Page](https://cvlab-unibo.github.io/inr2vec/)]

---
The code contained in this repository has been tested on Ubuntu 20.04 with Python 3.8.20.

## Setup
Create a virtual environment and install the library `pycarus`:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip setuptools
$ pip install pycarus
```
Then, try to import `pycarus` to get the command that you can run to install all the needed Pytorch libraries:
```
$ python3
>>> import pycarus
...
ModuleNotFoundError: PyTorch is not installed. Install it by running: source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
In this example, you can install all the needed Pytorch libraries by running:
```
$ source /XXX/.venv/lib/python3.8/site-packages/pycarus/install_torch.sh
```
This script downloads and installs the wheels for torch, torchvision, pytorch3d and torch-geometric.
Occasionally, it may fails due to pytorch3d wheel not being available anymore. If that happens,
please let us know or try to install pytorch3d manually.  
Finally install the other dependencies:
```
$ pip install hesiod torchmetrics wandb h5py==3.0.0
```
## Experiments
The code for each experiment has been organized in a separate directory, containing also a README file with all the instructions.  

## Datasets
Please contact Pierluigi Zama Ramirez (pierluigi.zama@unibo.it) if you need access to the datasets used in the experiments, both the ones containing the raw 3D shapes and the ones with the INRs.

## Cite us
If you find our work useful, please cite us:
```
@inproceedings{deluigi2023inr2vec,
    title = {Deep Learning on Implicit Neural Representations of Shapes},
    author = {De Luigi, Luca 
              and Cardace, Adriano 
              and Spezialetti, Riccardo 
              and Zama Ramirez, Pierluigi 
              and Salti, Samuele
              and Di Stefano, Luigi},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```
