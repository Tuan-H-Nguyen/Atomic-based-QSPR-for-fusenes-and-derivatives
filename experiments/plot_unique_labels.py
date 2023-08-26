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

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling, DEFAULT_PATH
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import data_splitter, graph_getter, Pipeline, RMSD, model_getter

from utils.plot_utility_v3 import scatter_plot

random_state = 2020
data_generator = ReducedData(
    N = 100, seed = random_state, path = DEFAULT_PATH,
    pah_only = False, subst_only = False)

elec_prop = ["BG"]
data = data_generator()

all_smiles = list(data.loc[:,"smiles"])
all_graphs = []
for smiles in all_smiles:
    all_graphs.append(
        smiles2graph(smiles,sp = True))

num_unique_label = {}
methods = [WLSubtree, WLEdge, WLShortestPath]
for l,label_method in enumerate(["WL-A","WL-AB","WL-AD"]):
    foo = []
    for i in range(0,6):

        graph_vectorizer = GraphVectorizer(
            label_method = methods[l], 
            num_iter = i, smiles = False)

        graph_vectorizer.fit(all_graphs)

        foo.append(len(graph_vectorizer.unique_labels))

    num_unique_label.update({
        label_method:np.array(foo)})

plot = scatter_plot()
for method,n_labels in num_unique_label.items():
    plot.add_plot(
        range(len(n_labels)),
        np.log10(n_labels),
        label = method,
        plot_line = True,
        xticks_format = 0,
        #yticks_format = 0
        xlabel = "Number of iterations",
        ylabel = "Base 10 logarithm of the length of $\phi$ vector"
        )

plot.ax.legend()
plot.save_fig("num_unique_labels.png",dpi =600)

