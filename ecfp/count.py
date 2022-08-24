import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
from zlib import crc32
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD

from pipeline import model_getter, graph_getter, data_splitter, ECFPVectorizer
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

N = 10
data = "mixed"
elec_prop_list = ["BG","EA","IP"]
num_iters = list(range(0,11,2))
random_state = 2022

if data == "mixed":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = False)
elif data == "pah":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = True, subst_only = False)
elif data == "subst":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = True)

data = data_generator()
graphs,_ = graph_getter(data)
smiles = data.loc[:,"smiles"]

def hash_(l):
    """
    Return an integer from a list by hashing it
    """
    l = list(l)
    strl = "".join([str(i) for i in l])
    hash_int = crc32(strl.encode("utf8")) & 0xffffffff
    return hash_int

def indexing(l):
    result = [] 
    for i,_ in enumerate(l):
        if i > 0:
            result.append(i)
    return result

for num_iter in range(0,8,1):
    print("num_iter = ",num_iter)
    ecfp_vectorizer = ECFPVectorizer(
        num_iter = num_iter, len_fp = 1129)

    wla_vectorizer = GraphVectorizer(
        label_method = WLSubtree, 
        num_iter = num_iter, smiles = False)

    wlab_vectorizer = GraphVectorizer(
        label_method = WLEdge, 
        num_iter = num_iter, smiles = False)

    wla = wla_vectorizer.fit(graphs)
    wla = wla_vectorizer.transform(graphs)
    unique_wla = len(list(set([hash_(w) for w in wla])))
    print(unique_wla/len(data))

    wlab = wlab_vectorizer.fit(graphs)
    wlab = wlab_vectorizer.transform(graphs)
    unique_wlab = len(list(set([hash_(w) for w in wlab])))
    print(unique_wlab/len(data))

    ecfp = ecfp_vectorizer.transform(smiles)
    unique_ecfp = len(list(set([hash_(e) for e in ecfp])))
    print(unique_ecfp/len(data))


