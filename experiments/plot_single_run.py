import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
path = "\\".join(path[:-1])
sys.path.append(path)
print(path)

import numpy as np

from data.data import ReducedData, stratified_sampling
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)
from pipeline import Pipeline, model_getter, graph_getter, data_splitter, RMSD

from utils.plot_utility_v3 import scatter_plot

random_state = 2020
data_type = "subst"

if data_type == "mixed":
    train_set,test_set = data_splitter(
        ReducedData(
            2000, random_state,
            path = path+"\\data",
            pah_only = False, subst_only = False),
        0.7,random_state)
elif data_type == "pah":
    train_set,test_set = data_splitter(
        ReducedData(
            2000, random_state,
            path = path+"\\data",
            pah_only = True, subst_only = False),
        0.7,random_state)
elif data_type == "subst":
    train_set,test_set = data_splitter(
        ReducedData(
            2000, random_state,
            path = path+"\\data",
            pah_only = False, subst_only = True),
        0.7,random_state)

train_graphs, test_graphs = graph_getter(train_set,test_set)

train_Y = np.array(train_set.loc[:,["BG"]])
test_Y = np.array(test_set.loc[:,["BG"]])

regressor, r_param_grid = model_getter("gpr")
model = Pipeline(
    vectorizing_method = WLSubtree,
    gv_param_grid = {"num_iter":[2,3]},
    regressor = regressor,
    r_param_grid = r_param_grid)

model.fit(train_graphs, train_Y)

test_Y_hat,test_std = model.predict(test_graphs,return_std = True)

plot_std = scatter_plot()

plot_std.add_plot(
    range(len(test_Y)),
    test_std)

plot_std.save_fig(path+"\\experiments\\[result2]\\std_"+data_type+".jpeg")
