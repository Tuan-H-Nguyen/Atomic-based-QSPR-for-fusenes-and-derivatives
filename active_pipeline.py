import time
import pickle
import random
from tqdm import tqdm
import numpy as np

from sklearn.base import BaseEstimator
#from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
from models.gpr import ModGaussianProcessRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles, MolToSmiles

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
#from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)
from pipeline import data_selector, data_splitter, model_getter, Pipeline, graph_getter, RMSD


kernel_str = input("input kernel type?")
if not kernel_str: kernel_str = "subtree"

data_type = input("input data type?")
if not data_type: data_type = "mixed"

repeat = input("number of repetitions?")
if not repeat: repeat = 10
else: repeat = int(repeat)

model, hyperp = model_getter("gpr")
if data_type == "subst":
    loops = 14
    steps = 31
    initial_split = 0.30
    num_iter = [3]
    #hyperp["alpha"].remove(5e-2)

elif data_type == "pah":
    loops = 15
    steps = 8
    initial_split = 0.30
    num_iter = [2,3]

elif data_type == "mixed":
    loops = 16
    steps = 13
    initial_split = 0.30
    num_iter = [2]
    #hyperp["alpha"].remove(5e-2)

def random_argmax(stds):
    max_std = np.max(stds)
    diff_std = abs(stds - max_std)
    diff_std = diff_std.reshape(-1)
    select = np.argwhere(diff_std < 1e-3)
    idx = random.choice(select)
    return idx[0]


result_dict = {
    "train_set_size":[],
    "active": [],
    "random": []
    }

data_generator = data_selector(data_type, "data",2020)
if kernel_str == "subtree":
    kernel = WLSubtree
elif kernel_str == "edge":
    kernel = WLEdge
else:
    raise Exception("")
###################################################################

for i in range(repeat):
    start = time.time()
    pipeline = Pipeline(
        vectorizing_method = kernel, 
        gv_param_grid = {"num_iter":num_iter},
        regressor = model,
        r_param_grid = hyperp,
        #input_smiles = True
        )

    train_set, test_set = data_splitter(
        data_generator, initial_split*0.7, 2020)

    train_graphs, test_graphs = graph_getter(train_set,test_set)

    RMSD_list = []
    train_len_list = []

    elec_props_list = ["BG","EA","IP"]
    train_Y = np.array(train_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))
    test_Y = np.array(test_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))

    for i in tqdm(range(loops)):

        # 1
        # train model
        pipeline.fit(train_graphs,train_Y)

        # 2
        # predict and calculate the standard deviation
        test_Y_hat,Y_std = pipeline.predict(
            test_graphs,return_std = True)
        # for each compound, take the maximum value of STD out of 
        # three values for three properties
        Y_std = np.max(Y_std,axis = 1)

        #store the RMSD and training set size
        rmsd = RMSD(test_Y_hat,test_Y)
        RMSD_list.append(rmsd)

        train_len_list.append(len(train_graphs))

        for i in range(steps):
            # choose one data points with maximum STD
            # if several compounds have similar STD, they are selected at random
            new_id = random_argmax(Y_std)

            # 3
            # remove the X of the selected point from the test set
            new_train_graph = test_graphs.pop(new_id)
            # add the X of the selected point from the training set
            train_graphs.append(new_train_graph)

            new_train_labels = test_Y[new_id,:][np.newaxis,:]
            # add the Y of the selected point from the training set
            train_Y = np.concatenate([train_Y,new_train_labels])
            # remove the Y of the selected point from the test set
            test_Y = np.delete(test_Y,new_id,0)

            Y_std = np.delete(Y_std ,new_id , 0)

    result_dict["train_set_size"] = train_len_list
    result_dict["active"].append(RMSD_list)

    print(len(train_graphs))
    print(len(test_graphs))
    print(time.time() - start)

###################################################################

for i in range(repeat):
    pipeline = Pipeline(
        vectorizing_method = kernel, 
        gv_param_grid = {"num_iter":num_iter},
        regressor = model,
        r_param_grid = hyperp,
        #input_smiles = True
        )

    train_set, test_set = data_splitter(
        data_generator, initial_split*0.7, 2020)

    train_graphs, test_graphs = graph_getter(train_set,test_set)

    RMSD_list = []

    train_Y = np.array(train_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))
    test_Y = np.array(test_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))

    for i in tqdm(range(loops)):

        pipeline.fit(train_graphs,train_Y)

        test_Y_hat,Y_std = pipeline.predict(
            test_graphs,return_std = True)

        RMSD_list.append(RMSD(test_Y_hat,test_Y))

        # same as above except the data points are chosen at random
        for i in range(steps):
            new_id = np.random.randint(0,len(Y_std))

            new_train_graph = test_graphs.pop(new_id)
            train_graphs.append(new_train_graph)

            new_train_labels = test_Y[new_id,:][np.newaxis,:]
            train_Y = np.concatenate([train_Y,new_train_labels])
            test_Y = np.delete(test_Y,new_id,0)

            Y_std = np.delete(Y_std ,new_id , 0)

    result_dict["random"].append(RMSD_list)


###################################################################

with open("experiments\\active_learning_"+kernel_str+"_"+data_type+".pkl","wb") as log:
    pickle.dump(result_dict,log)


