import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
from copy import deepcopy

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

from pipeline import data_splitter, graph_getter, vectorizing_data, model_getter

random_state = 2020
num_iter = 2
data_type = "subst"

if data_type == "mixed":
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = False, subst_only = False)
elif data_type == "pah":
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = True, subst_only = False)
elif data_type == "subst":
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = False, subst_only = True)

train_set,test_set = data_splitter(
    data_generator, train_split = 0.7, random_state = random_state)

train_graphs,test_graphs = graph_getter(train_set,test_set)

graph_vectorizer = GraphVectorizer( 
    graphs = train_graphs + test_graphs,
    label_method = WLShortestPath, num_iter = num_iter)

train_X = graph_vectorizer.bulk_vectorize(train_graphs)
test_X = graph_vectorizer.bulk_vectorize(test_graphs)

elec_prop = "BG"
train_Y = np.array(list(train_set.loc[:,elec_prop]))
test_Y = np.array(test_set.loc[:,elec_prop])

regressor = model_getter("rr")

regressor.fit(train_X,train_Y)

re_coef = regressor.best_estimator_.coef_
max_contribute = np.max(re_coef)

plot = scatter_plot()

plot.ax.hist(re_coef, 
    bins = [-i*0.00001 for i in range(0,400)][::-1]+[i*0.00001 for i in range(1,400)]
    )
plot.add_plot(
    [],[],
    xlabel = "Associated RR coefficients' values", 
    ylabel = "Number of labels",
    xticks_format = 3, yticks_format = 0
    )
    
plot.save_fig("\\".join(path) + "\\shortestPath_rr_coef_hist.jpeg",dpi=600)

explain_dict = dict(zip(
    graph_vectorizer.unique_labels,
    re_coef.tolist()))

sample = "c1cc2c(cc1)c1c(ccc(c1)C#N)c1c2cccc1"

node_feat, adj_list,edges_list, edges_feat, sp_dists = smiles2graph(sample,sp=True)

wl_labelling = WLShortestPath(node_feat, adj_list,edges_list, edges_feat, sp_dists)

for _ in range(num_iter):
    wl_labelling.relabelling_nodes()














