# Atomic-based QSPR for fusenes and derivatives.

## Introduction

This repository is dedicated to the upcoming paper on the use Weisfeiler-Lehman
kernel method to build Quantitative Structure-Property Relationship for
electronic properties of Polycylic Aromatic Hydrocarbon (PAH), thienoacenes,
and substituted PAH molecules.

## Data

The data stored in .csv files can be found within the data folder.

## Code

Code for Weisfeiler-Lehman algorithm and molecular representation can be found
in `wl\labelling_graph.py`.  
Code for the WL kernel pipeline can be found in the `pipeline.py`.   
All experiments data, code, and plots described in the manuscript can be found
within the `experiments` folder.

## Using predictor to generate prediction

First of all, train the model using `main.py`.

    python main.py -k wla -m gpr -d mixed -i 2 3 -n 10 -o 1 -p 1

where:
- `-k wla` : specified the kernel method, which is WL-A. Other keywords include
  subtree (alias for wla), wlab or edge, wlad or shortest\_path, ...
- `-m gpr` : specified the regressor, which is Gaussian Process Regressor. Other
  keyword: rr, gpr_ (for GPR with linear kernel)
- `-d mixed` : specified datasets, which is mixed datasets. Other kw: pah, subst.
- `-i 2 3` : specified number(s) of iterations. If multiple values are provided,
  grid search will be performed to find the most optimal
- `-n 10` : number of models for training and testing. Average train/test error
  will be provided. Multiple models are saved and used as ensemble for
  prediction.
- `-o 1` : specify 1 to saved error report. 0 for not saving anything
- `-p 1` : specify 1 to saved models in pickle file. 0 for not saving anything
  
With the last keywords, a pickle file with the name of
`model_ensemble_[dataset_name]_[kernel_type]_[regressor].pkl`, e.g.
`model_ensemble_mixed_subtree_gpr.pkl` will be generated. This file is used by
the preditor to generate prediction. The path to this file can be modified by
the variable `PATH_TO_PKL` in both `main.py` and `predictor.py`.  
Then, use predictor to generate prediction:

    python predictor.py [smiles string] -m gpr -k wla -d mixed

- `[smiles string]`: smiles string of the molecule whose prediction on electronic
  properties will be generated. Note: quotation marks (' or ") maybe needed.
- other parameters are similar as above.
