"""
Active learning experiment for WL-AD/shortest path kernel model
"""

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-2]))
sys.path.append("\\".join(path[:-2]))

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

#from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles, MolToSmiles

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
#from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)
from pipeline import data_selector, data_splitter, model_getter, Pipeline, graph_getter, RMSD

def random_argmax(stds):
    max_std = np.max(stds)
    diff_std = abs(stds - max_std)
    diff_std = diff_std.reshape(-1)
    select = np.argwhere(diff_std < 1e-3)
    idx = random.choice(select)
    return idx[0]

def learning(
    data_generator, 
    initial_split, final_split, random_state, steps,
    kernel, model, r_param_grid, num_iter,
    repeat ,active = True
    ):

    all_RMSD_list = []

    for run in range(repeat):
        start = time.time()
        pipeline = Pipeline(
            vectorizing_method = kernel, 
            gv_param_grid = {"num_iter":num_iter},
            regressor = model,
            r_param_grid = r_param_grid)

        train_set, test_set = data_splitter(
            data_generator, initial_split*final_split, random_state)

        train_graphs, test_graphs = graph_getter(train_set,test_set)

        final_train_size = int(final_split*(len(train_graphs)+len(test_graphs)))

        RMSD_list = []
        train_len_list = []

        elec_props_list = ["BG", "EA", "IP"]

        train_Y = np.array(train_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))
        test_Y = np.array(test_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))

        while len(train_graphs) < final_train_size:
            start = time.time()
            # 1 train model
            pipeline.fit(train_graphs,train_Y)

            # 2 predict and calculate the standard deviation
            test_Y_hat,Y_std = pipeline.predict(
                test_graphs,return_std = True)
            # for each compound, take the maximum value of STD out of 
            # three values for three properties
            Y_std = np.max(Y_std,axis = 1)

            #store the RMSD and training set size
            rmsd = RMSD(test_Y_hat,test_Y)
            RMSD_list.append(rmsd)

            train_len_list.append(len(train_graphs))

            sampling_steps = final_train_size - len(train_graphs)
            runtime = time.time() - start
            assert sampling_steps > 0
            print("Run {}/{}, {:0f} left. ETA:{:2f}s".format(
                run+1,repeat,sampling_steps/steps,sampling_steps*runtime/steps))

            sampling_steps = steps if steps < sampling_steps else sampling_steps

            for i in range(sampling_steps):
                # choose one data points with maximum STD
                # if several compounds have similar STD, they are selected at random
                if active:
                    new_id = random_argmax(Y_std)
                else:
                    new_id = np.random.randint(0,len(Y_std))

                # 3 remove the X of the selected point from the test set
                new_train_graph = test_graphs.pop(new_id)
                # add the X of the selected point from the training set
                train_graphs.append(new_train_graph)

                new_train_labels = test_Y[new_id,:][np.newaxis,:]
                # add the Y of the selected point from the training set
                train_Y = np.concatenate([train_Y,new_train_labels])
                # remove the Y of the selected point from the test set
                test_Y = np.delete(test_Y,new_id,0)

                Y_std = np.delete(Y_std ,new_id , 0)

        RMSD_list = np.vstack(RMSD_list)
        #print(RMSD_list)
        all_RMSD_list.append(RMSD_list)

    return train_len_list, all_RMSD_list

def active_learning_pkl(kernel_str, data_type, repeat, random_state):
    start = time.time()
    result_dict = {
        "train_set_size":[],
        "active": [],
        "random": []
        }

    model, hyperp = model_getter("gpr_")
    initial_split = 0.2
    final_split = 0.7
    number_sampling = 20
    if data_type == "subst":
        steps = int( 887*final_split*(1-initial_split) / number_sampling )
        num_iter = [0,1,2]

    elif data_type == "pah":
        steps = int( 248*final_split*(1-initial_split) / number_sampling )
        num_iter = [0,1,2]

    elif data_type == "mixed":
        steps = int( 425*final_split*(1-initial_split) / number_sampling )
        num_iter = [0,1]

    data_generator = data_selector(data_type, "data",random_state)
    if kernel_str == "shortest_path":
        kernel = WLShortestPath
    else:
        raise Exception("")
    ###################################################################
    train_len_list, RMSD_list = learning(
        data_generator,initial_split,final_split, random_state = random_state,
        kernel = kernel, model = model, r_param_grid = hyperp, num_iter = num_iter,
        repeat = repeat, steps = steps, 
        active = True)
    result_dict["train_set_size"] = train_len_list
    result_dict["active"] = RMSD_list

    print(time.time() - start)

    ###################################################################

    train_set_size, RMSD_list = learning(
        data_generator,initial_split, final_split,random_state = random_state,
        kernel = kernel, model = model, r_param_grid = hyperp ,num_iter = num_iter,
        repeat = repeat, steps = steps, 
        active = False)
    result_dict["random"] = RMSD_list

    ###################################################################

    pkl_path = "active_learning_"+kernel_str+"_"+data_type+"_" +str(random_state) +".pkl"

    print(pkl_path)
    with open(pkl_path,"wb") as log:
        pickle.dump(result_dict,log)

repeat = input("number of repetitions?")
if not repeat: repeat = 1
else: repeat = int(repeat)

kernel_str = input("kernel name?")
data_type = input("data name?")
random_state = int(input("random state?"))

if not kernel_str and not data_type:
    print("assess active learning for all models on all dataset")
    for kernel_str in ["shortest_path"]:
        print("kernel_str: ", kernel_str)
        for data_type in ["mixed","pah","subst"]:
            print("data_type: ", data_type)

            active_learning_pkl(kernel_str,data_type,repeat,random_state)

else:
    active_learning_pkl(kernel_str,data_type,repeat,random_state)
