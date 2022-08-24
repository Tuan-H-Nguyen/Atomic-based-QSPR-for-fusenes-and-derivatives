import sys, os
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

from pipeline import model_getter, graph_getter, data_splitter

data = "subst"
N = 10
random_state = 2022

num_iters = [0,1,2,3,4]
elec_prop_list = ["BG","EA","IP"]

method_list = [
    (WLSubtree,"rr"),(WLSubtree,"gpr"),
    (WLEdge,"rr"), (WLEdge,"gpr"),
    #(WLShortestPath,"rr"),(WLShortestPath,"krr")
    ]

result = np.zeros((
    N,len(method_list)*len(elec_prop_list),
    len(num_iters)))

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

def vectorizing_data(
    train_graphs, test_graphs, 
    vectorizing_method,
    num_iter, graph_vectorizer = None):

    """
    #################
    Vectorizing Graph
    #################
    """
    start = time.time()

    if graph_vectorizer == None:
        graph_vectorizer = GraphVectorizer( 
            graphs = train_graphs,
            label_method = vectorizing_method, num_iter = num_iter)

    train_X = graph_vectorizer.bulk_vectorize(train_graphs)
    test_X = graph_vectorizer.bulk_vectorize(test_graphs)

    print("Vectorizing runtime", time.time() - start)
    print("\tNumber of unique labels: ", len(graph_vectorizer.unique_labels))

    return train_X, test_X

for i in range(N):
    print("Run #: {}".format(i))

    train_set, test_set = data_splitter(
        data_generator, train_split = 0.7, random_state = random_state)

    for j, method in enumerate(method_list):

        for num_iter in num_iters:
            label_method, regress_model = method

            if j > 3 and ((num_iter >= 3 and data != "pah") or (num_iter >= 4 and data == "pah")):
                continue

            train_graphs,test_graphs = graph_getter(
                train_set,test_set, sp = True)

            train_X,test_X = vectorizing_data(
                train_graphs, test_graphs,
                wl_labelling_method = label_method,
                num_iter = num_iter)

            for e,elec_prop in enumerate(elec_prop_list):
                train_Y = np.array(list(train_set.loc[:,elec_prop]))
                test_Y = np.array(test_set.loc[:,elec_prop])

                regressor = model_getter(regress_model)

                regressor.fit(train_X,train_Y)

                test_Y_hat = regressor.predict(test_X)
                test_rmsd = RMSD(test_Y_hat, test_Y)

                print(e+j*len(elec_prop_list))

                result[i,e+j*len(elec_prop_list), num_iter] += test_rmsd 

with open("\\".join(path) + "\\vs_num_iter_"+data+".pkl","wb") as handle:
    pickle.dump(result,handle)



