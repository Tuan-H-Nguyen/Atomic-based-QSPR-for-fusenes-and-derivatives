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

def random_argmax(stds):
    max_std = np.max(stds)
    diff_std = abs(stds - max_std)
    diff_std = diff_std.reshape(-1)
    select = np.argwhere(diff_std < 1e-3)
    idx = random.choice(select)
    return idx[0]

def learning(
    data_generator, initial_split, 
    kernel, num_iter, 
    model, r_param_grid,
    steps, loops, repeat ,
    active = True):

    all_RMSD_list = []

    for i in range(repeat):
        start = time.time()
        pipeline = Pipeline(
            vectorizing_method = kernel, 
            gv_param_grid = {"num_iter":num_iter},
            regressor = model,
            r_param_grid = r_param_grid)

        train_set, test_set = data_splitter(
            data_generator, initial_split, 2020)

        train_graphs, test_graphs = graph_getter(train_set,test_set)

        RMSD_list = []
        train_len_list = []

        elec_props_list = ["BG","EA","IP"]
        train_Y = np.array(train_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))
        test_Y = np.array(test_set.loc[:,elec_props_list]).reshape(-1,len(elec_props_list))

        for i in tqdm(range(loops)):
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

            for i in range(steps):
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

        all_RMSD_list.append(RMSD_list)

    return train_len_list, all_RMSD_list

def active_learning_pkl(kernel_str, data_type, repeat):
    start = time.time()
    result_dict = {
        "train_set_size":[],
        "active": [],
        "random": []
        }

    model, hyperp = model_getter("gpr")
    initial_split = 0.3
    hyperp["alpha"].remove(5e-2)
    if data_type == "subst":
        loops = 31
        num_iter = [2]
        steps = 14
        if kernel_str == "subtree":
            pass
        elif kernel_str == "edge":
            #initial_split = 0.64
            steps = 28
            loops = 16
            num_iter = [1,2]

    elif data_type == "pah":
        loops = 30
        num_iter = [2,3]
        steps = 4
        if kernel_str == "subtree":
            pass
        elif kernel_str == "edge":
            #initial_split = 0.64
            #steps = 2
            num_iter = [1,2,3]

    elif data_type == "mixed":
        loops = 50
        num_iter = [2]
        initial_split = 0.45
        steps = 4
        if kernel_str == "subtree":
            pass
        elif kernel_str == "edge":
            #initial_split = 0.7
            #steps = 2
            num_iter = [0,1,2]

    data_generator = data_selector(data_type, "data",2020)
    if kernel_str == "subtree":
        kernel = WLSubtree
    elif kernel_str == "edge":
        kernel = WLEdge
    else:
        raise Exception("")
    ###################################################################
    train_len_list, RMSD_list = learning(
        data_generator,0.7*initial_split, 
        kernel = kernel, num_iter = num_iter,
        model = model, r_param_grid = hyperp,
        repeat = repeat, steps = steps, loops = loops, 
        active = True)
    result_dict["train_set_size"] = train_len_list
    result_dict["active"].append(RMSD_list)

    print(time.time() - start)

    ###################################################################

    train_set_size, RMSD_list = learning(
        data_generator,0.7*initial_split, 
        kernel = kernel, num_iter = num_iter,
        model = model, r_param_grid = hyperp ,
        repeat = repeat, steps = steps, loops = loops,
        active = False)
    result_dict["random"].append(RMSD_list)

    ###################################################################

    with open("experiments\\active_learning_"+kernel_str+"_"+data_type+".pkl","wb") as log:
        pickle.dump(result_dict,log)

repeat = input("number of repetitions?")
if not repeat: repeat = 1
else: repeat = int(repeat)

kernel_str = input("kernel name?")
data_type = input("data name?")

if not kernel_str and not data_type:
    print("assess active learning for all models on all dataset")
    for kernel_str in ["subtree","edge"]:
        print("kernel_str: ", kernel_str)
        for data_type in ["mixed","pah","subst"]:
            print("data_type: ", data_type)

            active_learning_pkl(kernel_str,data_type,repeat)

else:
    active_learning_pkl(kernel_str,data_type,repeat)
