import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import model_getter, graph_getter, vectorizing_data, data_splitter

random_state = 2022

for data_type in ["mixed","pah","subst"]:
    N = 10

    elec_prop_list = ["BG","EA","IP"]
    method_list = [
        "subtree-rr", "subtree-krr",
        "edge-rr", "edge-krr",
        "shortest_path-rr","shortest_path-krr"]

    with open("\\".join(path) + "\\vs_num_iter_"+ data_type+".pkl","rb") as handle:
        result = pickle.load(handle)
    std = np.std(result,axis= 0)
    result = np.mean(np.array(result),axis=0)
    result[result == 0.] = 100.

    num_iters = []
    for i in range(0,result.shape[0], 3):
        num_iters.append(np.argmin(np.sum(result[i:i+3,:],axis=0)))

    print(data_type)
    for j,method in enumerate(method_list):
        print(method)
        for e,elec_prop in enumerate(elec_prop_list):
            print("\t#", elec_prop)
            n = num_iters[j]
            print("\t\tNum iter: ", n)
            print("\t\tRMSD : ",result[e+j*len(elec_prop_list),n])
            print("\t\tSTD : ",std[e+j*len(elec_prop_list),n])




