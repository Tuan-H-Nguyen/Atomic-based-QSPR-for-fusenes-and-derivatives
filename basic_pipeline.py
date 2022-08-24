import time
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles, MolToSmiles

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

def data_splitter(data_generator,train_split,random_state,fixed_test_size=False):
    """
    ################
    Data preparation
    ################
    """
    data = data_generator()

    train_set = stratified_sampling(
        int(train_split*len(data)),data, random_state = random_state)

    test_set = data.drop(train_set.index, axis = 0)

    if fixed_test_size:
        if fixed_test_size < 1.0:
            assert isinstance(fixed_test_size,float)
            test_set = stratified_sampling(
                int(fixed_test_size*len(data)),test_set, random_state = random_state)
        else:
            assert isinstance(fixed_test_size,int) and fixed_test_size > 0
            test_set = stratified_sampling(
                fixed_test_size,test_set, random_state = random_state)

    print("\tNumber of training instances: ", len(train_set))
    print("\tNumber of test instances: ", len(test_set))

    return train_set, test_set

def model_getter(model):
    if model == "krr":
        regressor = GridSearchCV(
            KernelRidge(), param_grid = {
                "kernel": ["rbf"],
                "alpha": [1e-3,1e-2],
                "gamma":[1e-5,1e-3]
                })
    elif model == "krr_poly":
        regressor = GridSearchCV(
            KernelRidge(), param_grid = {
                "kernel": ["polynomial"],
                "alpha": [10e-2, 10e-1, 1],
                "degree":[0,1,2,3,4],
                })

    elif model == "rr":
        regressor = GridSearchCV(
            Ridge(), param_grid = {
                "alpha": [10e-3,10e-2, 10e-1, 1],
                })

    else:
        raise Exception("Model {} has not been implemented".format(args.model))

    return regressor

def graph_getter(train_set, test_set = None, sp = True):

    """
    #######################
    Convert SMILES to graph
    #######################
    """
    start = time.time()

    train_graphs = smiles2graph(list(train_set.loc[:,"smiles"]),sp = sp)

    if test_set is not None:
        test_graphs = smiles2graph(list(test_set.loc[:,"smiles"]),sp = sp)
    else:
        test_graphs = None

    print("Transform to graph runtime", time.time() - start)

    return train_graphs, test_graphs

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
            label_method = vectorizing_method, num_iter = num_iter)

    graph_vectorizer.fit(train_graphs)

    train_X = graph_vectorizer.transform(train_graphs)
    test_X = graph_vectorizer.transform(test_graphs)

    print("Vectorizing runtime", time.time() - start)
    print("\tNumber of unique labels: ", len(graph_vectorizer.unique_labels))

    return train_X, test_X

def dim_reducer_getter(train_X,train_Y,svd_factor):
    start = time.time()

    len_vec = train_X.shape[1]

    dim_reducer = TruncatedSVD(
        int(len_vec/svd_factor),
        algorithm  = "randomized")

    train_X_ = dim_reducer.fit_transform(train_X, train_Y)

    print("\tReducing dimension runtime", time.time() - start)

    return dim_reducer, train_X_

def ecfp_fingerprint(data_set,radius=2,nbits = 2048):
    smiles_list = list(data_set.loc[:,"smiles"])
    X = [] 
    for i,smiles in enumerate(smiles_list):
        smiles = MolFromSmiles(smiles)
        X.append(np.array(GetMorganFingerprintAsBitVect(smiles,radius,nBits= nbits)))
    return X

def main_pipeline(
    model, 
    data_generator,
    vectorizing_method,
    num_iter,
    train_split,
    svd_factor,
    random_state
    ):

    train_set,test_set = data_splitter(data_generator,train_split,random_state)
    """
    ######################
    Vectorizing graph data
    ######################
    """
    if vectorizing_method == "ecfp":
        train_X = ecfp_fingerprint(train_set, num_iter, 4096)
        test_X = ecfp_fingerprint(test_set, num_iter, 4096)
    else:
        train_graphs,test_graphs = graph_getter(
            train_set,test_set, sp = True if vectorizing_method == WLShortestPath else False)

        train_X, test_X = vectorizing_data(
            train_graphs,test_graphs,
            vectorizing_method,
            num_iter)

    #######################

    all_train_rmsd = []
    all_test_rmsd = []
    for elec_prop in ["BG","EA","IP"]:
        """
        ##########################
        Get the true target values
        ##########################
        """

        train_Y = np.array(list(train_set.loc[:,elec_prop]))
        test_Y = np.array(test_set.loc[:,elec_prop])

        """
        #################
        Reducing Features
        #################
        """
        if svd_factor > 1:
            dim_reducer, train_X_ = dim_reducer_getter(
                train_X, train_Y, svd_factor)
        else:
            train_X_ = train_X

        """
        ##################
        Building regressor
        ##################
        """
        start = time.time()

        regressor = model_getter(model)
        regressor.fit(
            train_X_, 
            train_Y
            )

        print("Model detail: ",regressor.best_estimator_)
        print("\tFit model runtime", time.time() - start)

        """
        ################
        Evaluating model
        ################
        """

        train_Y_hat = regressor.predict(train_X_)
        train_rmsd = RMSD(train_Y_hat, train_Y)

        if svd_factor > 1:
            test_X_ = dim_reducer.transform(test_X)
        else:
            test_X_ = test_X

        test_Y_hat = regressor.predict(test_X_)
        test_rmsd = RMSD(test_Y_hat, test_Y)

        print("""
            ##########
            ### {} ###
            ##########
            """.format(elec_prop)
            )
        print("Train RMSD: {:.3f}eV".format(train_rmsd))
        print("Test RMSD: {:.3f}eV".format(test_rmsd))

        all_train_rmsd.append(train_rmsd)
        all_test_rmsd.append(test_rmsd)

    return all_train_rmsd + all_test_rmsd
