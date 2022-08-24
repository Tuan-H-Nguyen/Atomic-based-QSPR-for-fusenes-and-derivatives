import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from data.data import ReducedData, stratified_sampling
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import model_getter, graph_getter, Pipeline, data_splitter, RMSD

data_type = "pah"
random_state = 10 # 2022

N = 5

start = time.time()

train_ratio_list = [i*0.01*0.7 for i in range(10,101,15)]

with open("\\".join(path) + "\\vs_num_iter_"+ data_type+".pkl","rb") as handle:
    result = pickle.load(handle)
result = np.mean(np.array(result),axis=0)
result[result == 0.] = 100.

num_iters = []
for i in range(0,result.shape[0], 3):
    num_iters.append(np.argmin(np.sum(result[i:i+3,:],axis=0)))
        
elec_prop_list = ["BG","EA","IP"]

method_list = [
    (WLSubtree,"rr"),(WLSubtree,"krr"),
    (WLEdge,"rr"), (WLEdge,"krr"),
    (WLShortestPath,"rr"),(WLShortestPath,"krr")
    ]

if data_type == "pah":
    num_iters = [
        [3,4], [2,3],
        [3,4], [2,3],
        [2,3], [1,2,3]
        ]
else:
    num_iters = [
        [3,4], [2,3],
        [3,4], [2,3],
        [1,2], [1,2]
        ]

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

data_len = len(data_generator())

data_len = [int(ratio*data_len) for ratio in train_ratio_list]

with open("\\".join(path) + "\\vs_train_size_len_"+data_type+".pkl","wb") as handle:
    pickle.dump(data_len,handle)

result = np.zeros((
    N,len(method_list)*len(elec_prop_list),
    len(train_ratio_list)))

for i in range(N):
    print("Run #: {}".format(i))
    for j, method in enumerate(method_list):

        vectorizing_method, regress_model = method

        data = data_generator()
        all_graphs, _ = graph_getter(data,sp=True)

        for k,ratio in enumerate(train_ratio_list):
            train_set, test_set = data_splitter(
                data_generator, train_split = ratio, fixed_test_size = 0.3,
                random_state = random_state)

            train_graphs,test_graphs = graph_getter(train_set,test_set,sp=True)

            train_Y = np.array(train_set.loc[:,elec_prop_list])
            test_Y = np.array(test_set.loc[:,elec_prop_list])

            regressor,param_grid = model_getter(regress_model,vectorizing_method ,False)

            pipeline = Pipeline(
                vectorizing_method = vectorizing_method,
                gv_param_grid = {"num_iter" : num_iters[j]},
                regressor = regressor, r_param_grid = param_grid, input_smiles = False)

            pipeline.fit(train_graphs,train_Y)

            test_Y_hat = pipeline.predict(test_graphs)
            test_rmsd = RMSD(test_Y_hat, test_Y)

            n_prop = len(elec_prop_list)
            result[i,0+j*n_prop : n_prop+j*n_prop, k] += test_rmsd 

print("Total runtime: ", time.time() - start)
print(result)

with open("\\".join(path) + "\\vs_train_size_"+data_type+".pkl","wb") as handle:
    pickle.dump(result,handle)




