import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
from copy import deepcopy
import pickle
import itertools, collections
from tqdm import tqdm

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from models.gpr import ModGaussianProcessRegressor

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from utils.plot_utility_v3 import scatter_plot, font_legend

from pipeline import data_splitter, graph_getter, Pipeline, RMSD, model_getter

random_state = 2020

def ranking(test,number_top,reverse = False):
    rank = []
    idx_list = list(range(len(test)))
    idx_list = sorted(
        idx_list,
        key = lambda i: test[i],
        reverse = reverse
        )
    return idx_list[:number_top]

"""
try:
    with open("\\".join(path)+"\\error_analysis_"+data_type+".pkl","rb") as handle:
        top_rmsd = pickle.load(handle)
except FileNotFoundError:
"""

data_generator = ReducedData(
    N = 100, seed = random_state, path = "data",
    pah_only = False, subst_only = False)

elec_prop = ["BG"]
data = data_generator()

all_graphs = []
for smiles in data.loc[:,"smiles"]:
    all_graphs.append(
        smiles2graph(smiles))
all_Y = np.array(data.loc[:,elec_prop])

regressor,r_param_grid = model_getter("gpr")
model = Pipeline(
    vectorizing_method = WLSubtree, 
    gv_param_grid = {"num_iter":[2,3]},
    regressor = regressor,
    r_param_grid = r_param_grid,
    )

LOO_error = []

for i in tqdm(range(len(data))):
    train_X = all_graphs[:i] + all_graphs[i+1:]
    train_Y = np.concatenate(
        [all_Y[:i,:], all_Y[i+1:,:]])

    model.fit(train_X,train_Y)

    test_X = [all_graphs[i]]
    test_Y_hat = model.predict(test_X)

    rmsd = np.sqrt(
        (test_Y_hat[0][0] - all_Y[i])**2)
    LOO_error.append(rmsd)

with open("LOO_error.pkl","wb") as handle:
    pickle.dump(LOO_error, handle)
