import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

import numpy as np
from sklearn.decomposition import PCA

from data.data import ReducedData, stratified_sampling
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import model_getter, graph_getter, Pipeline, data_splitter, RMSD

from utils.plot_utility_v3 import scatter_plot

data_type = "subst"
random_state = 10 # 2022

if data_type == "mixed":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = False)
elif data_type == "pah":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = True, subst_only = False)
elif data_type == "subst":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = True)

for i,method in enumerate([WLSubtree,WLEdge,WLShortestPath]):
    for num_iter in [0,1,2]:
        graph_vectorizer = graph_vectorizer = GraphVectorizer(
            label_method = method, num_iter = num_iter, smiles = True)

        data_set = data_generator()

        smiles = list(data_set.loc[:,"smiles"])

        graph_vectorizer.fit(smiles)
        X = graph_vectorizer.transform(smiles)

        pca = PCA(n_components = 2)

        X_ = pca.fit_transform(X)

        plot = scatter_plot()

        Y = list(data_set.loc[:,"BG"])
        min_Y = min(Y)
        max_Y = max(Y)
        color = [str((y-min_Y)/(max_Y-min_Y)) for y in Y]

        plot.add_plot(
            X_[:,0],X_[:,1], scatter_color = color)

        if i == 0:
            plot.save_fig("pca_plots\\pca_plot_"+data_type+"_WLA_"+str(num_iter)+".jpeg",dpi =600)
        elif i == 1:
            plot.save_fig("pca_plots\\pca_plot_"+data_type+"_WLAB_"+str(num_iter)+".jpeg",dpi =600)
        elif i == 2:
            plot.save_fig("pca_plots\\pca_plot_"+data_type+"_WLAD_"+str(num_iter)+".jpeg",dpi =600)

