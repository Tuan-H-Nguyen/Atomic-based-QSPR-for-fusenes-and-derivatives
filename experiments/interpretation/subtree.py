import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-2]))
sys.path.append("\\".join(path[:-2]))
from copy import deepcopy
import pickle

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from utils.plot_utility_v3 import scatter_plot, font_legend

from pipeline import data_splitter, graph_getter, Pipeline, RMSD

num_iter = 5
data_type = "pah"
random_state = 2020

result_dict = {}
if data_type == "pah":
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = True, subst_only = False)
elif data_type == "subst":
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = False, subst_only = True)
else: raise Exception("")

train_set,test_set = data_splitter(
    data_generator, train_split = 0.7, random_state = random_state)

train_set.loc[:,"smiles"] =  train_set.loc[:,"smiles"].apply(
    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
test_set.loc[:,"smiles"] =  test_set.loc[:,"smiles"].apply(
    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))

train_graphs,test_graphs = graph_getter(train_set,test_set)

model = Pipeline(
    vectorizing_method = WLSubtree, 
    gv_param_grid = {"num_iter":[num_iter]},
    regressor = Ridge,
    r_param_grid = {"alpha":[1e-2,0.1]}
    )

### a for loop was initially here ###
elec_prop = ["BG"]

train_Y = np.array(train_set.loc[:,elec_prop])
test_Y = np.array(test_set.loc[:,elec_prop])

model.fit(train_graphs,train_Y)

re_coef = model.regressors[0].coef_ # 0 for BG

Y_test_hat = model.predict(test_graphs).reshape(-1)
test_rmsd =  np.sqrt(
    (Y_test_hat - np.array(list(test_set.loc[:,"BG"])).reshape(-1))**2
    )

explain_dict = dict(zip(
    model.graph_vectorizers[0].unique_labels, #0 to get model for BG
    re_coef.tolist()))

test_smiles = list(test_set.loc[:,"smiles"])

contributions_list = []
unknown_label = 0

for sample_no,sample in enumerate(test_smiles):
    #if sample_no > 100: break

    node_feat, adj_list,edges_list, edges_feat, sp_dists = smiles2graph(sample,sp = True)

    wl_labelling = WLSubtree(node_feat, adj_list,edges_list, edges_feat, sp_dists)

    for _ in range(num_iter):
        wl_labelling.relabelling_nodes()

    contributions = np.zeros(len(node_feat))
    """
    The list of atoms' labels are according to the order of node_feat, which is generated
    according to the order of mol.GetAtoms()
    Therefore, the contributions list is the contribution of atoms (inferred via the linear 
    regression coef) in the above order.
    """
    for label_set in wl_labelling.atom_labels:
        for i,label in enumerate(label_set):
            try:
                """
                contributions += np.array(
                    [explain_dict[label] for label in label_set])
                """
                contributions[i] += explain_dict[label]
            except KeyError:
                unknown_label += 1
                contributions[i] += 0

    contributions = list(contributions)
    contributions_list.append(contributions)

result_dict.update({
    "contributions_list":contributions_list,
    "test_Y":test_Y,
    "test_rmsd":test_rmsd,
    "test_smiles":test_smiles
    })

with open("\\".join(path) + "\\"+data_type+"\\result.pkl","wb") as handle:
    pickle.dump(result_dict, handle)
