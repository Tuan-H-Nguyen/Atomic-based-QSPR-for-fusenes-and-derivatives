import time
import numpy as np

from sklearn.base import BaseEstimator
#from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from models.gpr import ModGaussianProcessRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles, MolToSmiles

from molecular_graph.smiles import smiles2graph
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

try:
    from data.data import ReducedData, stratified_sampling
except ModuleNotFoundError:
    pass

def RMSD(Y,Y_hat):
    SE = (Y-Y_hat)**2
    MSE = np.mean(SE,axis = 0)
    return np.sqrt(MSE)

def data_selector(data,path,random_state):
    if data == "mixed":
        data_generator = ReducedData(
            N = 1000, seed = random_state, path = path,
            pah_only = False, subst_only = False)
    elif data == "pah":
        data_generator = ReducedData(
            N = 1000, seed = random_state, path = path,
            pah_only = True, subst_only = False)
    elif data == "subst":
        data_generator = ReducedData(
            N = 1000, seed = random_state, path = path,
            pah_only = False, subst_only = True)
    return data_generator

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

def model_getter(model,kernel = None,grid_search = False,name = None):
    name = "" if name == None else name+"__"
    if kernel == "shortest_path": gamma = [1e-6,10e-5,10e-4,10e-3]
    elif kernel == "subtree" or kernel == "edge": gamma = [10e-3]
    else: gamma = [10e-3,10e-5]
    if model == "krr":
        regressor,param_grid = KernelRidge, {
                name + "kernel": ["rbf"],
                name + "alpha": [1e-3,10e-2,1],
                name + "gamma": gamma
                }
    elif model == "rr":
        regressor,param_grid = Ridge, {
                name + "alpha": [10e-3, 0.1],
                }

    elif model =="gpr":
        #kernel1 = DotProduct(sigma_0 = 1,sigma_0_bounds = (1e-10,1e5)) * ConstantKernel(1.0)
        kernel2 = RBF() * ConstantKernel(1.0)
        regressor,param_grid = ModGaussianProcessRegressor, {
                "kernel": [kernel2] ,
                "alpha" : [5e-3],
                "_max_iter":[20000]
                }
    else:
        raise Exception("Model {} has not been implemented".format(args.model))

    if grid_search:
        return GridSearchCV(regressor(),param_grid=param_grid)
    else:
        return regressor, param_grid

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

class ECFPVectorizer(BaseEstimator):
    def __init__(self,num_iter=None,len_fp = 2048):
        self.num_iter = num_iter
        self.len_fp = len_fp

    def fit(self,X,Y=None):
        return self

    def transform(self,X,Y=None):
        X_out = [] 
        for i,smiles in enumerate(X):
            smiles = MolFromSmiles(smiles)
            X_out.append(np.array(GetMorganFingerprintAsBitVect(
                smiles, radius = self.num_iter, nBits= self.len_fp)))
        return X_out

class Pipeline:
    def __init__(
        self,
        vectorizing_method, gv_param_grid,
        regressor, r_param_grid,
        input_smiles = False
        ):
        """
        Pipeline for WL-based graph model that take in either SMILES strings and return prediction.
        Args:
        + vectorizing_method (class, either WLSubtree, WLEdge, or WLShortestPath): 
        + gv_param_grid (dict): dictionary in which keys are strings of parameters' names 
            and values are lists of candidate values for that parameters. E.g. {"num_iter":[2,3]}
        + regressor (class): object of sklearn regressor algorithm such as GaussianProcessRegressor
        + r_param_grid (dict):dictionary in which keys are strings of parameters' names 
            and values are lists of candidate values for that parameters. E.g. {"alpha":[1e-2,1e-3]}
        + input_smiles (boolean): whether input of the pipeline are SMILES string or graph (list of 
            nodes, adjacency matrices, ... )
        """

        assert vectorizing_method in [WLSubtree, WLEdge, WLShortestPath, "ecfp"]

        self.vectorizing_method = vectorizing_method
        self.gv_param_list = [ # list of graph vectorizer's hyperparameters 
            dict(zip(gv_param_grid,t)) for t in zip(*gv_param_grid.values())]

        self.regressor_class = regressor
        self.r_param_grid = r_param_grid

        self.input_smiles = input_smiles

    def fit(self,X,Y):
        """
        Fit the pipeline to X and Y of training data.
        Note that hyperparameters for both the WL graph vectorizer (e.g. num_iter)
        and the regressor are searched via grid_search.
        Args:
        + X (list): list of SMILES string or graphs
        + Y (array of shape [n_samples, n_properties]): 
        """
        n_prop = Y.shape[1]
        self.n_prop = n_prop

        error_log = np.zeros((len(self.gv_param_list),n_prop))
        regressor_log = []
        graph_vec_log = []

        for j,gv_param in enumerate(self.gv_param_list):
            """Iterate through the parameters grid of the vectorizer"""

            if self.vectorizing_method == "ecfp":
                graph_vectorizer = ECFPVectorizer(
                    **gv_param)

            else:
                graph_vectorizer = GraphVectorizer(
                    label_method = self.vectorizing_method,
                    smiles = self.input_smiles, **gv_param)

            graph_vec_log.append(graph_vectorizer)

            #fit transform X_train
            graph_vectorizer.fit(X)
            X_ = self.features(graph_vectorizer.transform(X))

            regressor_log_ = []
            for i in range(n_prop):
                """Iterate through the target properties"""
                Y_ = self.labels(Y[:,i])
                regressor = GridSearchCV(
                    self.regressor_class(),self.r_param_grid,
                    )

                #regressor.fit(X_,Y_)
                regressor = self.fit_model(regressor,X_,Y_)

                error_log[j,i] += regressor.best_score_

                regressor_log_.append(regressor.best_estimator_)

            regressor_log.append(regressor_log_)

        self.regressors = []
        self.graph_vectorizers = []

        for i in range(Y.shape[1]):
            """select best pipeline (graph vectorizer + regressor) for each property"""
            best_error_idx = np.argmax(error_log[:,i])
            self.regressors.append(regressor_log[best_error_idx][i])
            self.graph_vectorizers.append(graph_vec_log[best_error_idx])

    def fit_model(self,model,X,Y):
        model = model.fit(X,Y)
        return model

    def features(self,X):
        return X

    def labels(self,Y):
        return Y

    def print_best_estimator(self):
        if self.vectorizing_method == "ecfp":
            for i,graph_vectorizer in enumerate(self.graph_vectorizers):
                print("ECFP; ","(num_iter=",graph_vectorizer.num_iter,")")
                print(self.regressors[i])
                #print(self.regressors[i].best_estimator_)
        else:
            for i,graph_vectorizer in enumerate(self.graph_vectorizers):
                print(graph_vectorizer.label_method.__name__,"(num_iter=",graph_vectorizer.num_iter,")")
                print(len(graph_vectorizer.unique_labels))
                print(self.regressors[i])
                #print(self.regressors[i].best_estimator_)

    def predict(self,X,return_std = False):
        if return_std:
            Y = []
            Y_std = []
            for i in range(self.n_prop):
                X_ = self.graph_vectorizers[i].transform(X)
                y,y_std = self.regressors[i].predict(X_,return_std = True)
                Y.append(y)
                Y_std.append(y_std)
            return np.array(Y).T, np.array(Y_std).T
        else:
            Y = []
            for i in range(self.n_prop):
                X_ = self.graph_vectorizers[i].transform(X)
                Y.append(self.regressors[i].predict(X_))
            return np.array(Y).T

def main_pipeline(
    model, 
    data_generator,
    vectorizing_method,
    num_iter,
    train_split,
    random_state,
    elec_prop_list = ["BG","EA","IP"]
    ):

    train_set,test_set = data_splitter(data_generator,train_split,random_state)
    """
    ######################
    Vectorizing graph data
    ######################
    """
    if vectorizing_method == "ecfp":
        train_graphs = list(train_set.loc[:,"smiles"])
        test_graphs = list(test_set.loc[:,"smiles"])
    else:
        train_graphs, test_graphs = graph_getter(train_set,test_set)

    all_train_rmsd = []
    all_test_rmsd = []
    """
    ##########################
    Get the true target values
    ##########################
    """

    train_Y = np.array(train_set.loc[:,elec_prop_list])
    test_Y = np.array(test_set.loc[:,elec_prop_list])
    if len(elec_prop_list) == 1:
        train_Y, test_Y = train_Y.T, test_Y.T

    """
    ##################
    Building regressor
    ##################
    """
    start = time.time()

    regressor,param_grid = model_getter(model,vectorizing_method ,False)

    pipeline = Pipeline(
        vectorizing_method = vectorizing_method,
        gv_param_grid = {"num_iter" : num_iter},
        regressor = regressor, r_param_grid = param_grid, input_smiles = False)

    pipeline.fit(
        train_graphs, 
        train_Y
        )

    print("Model detail: ")
    pipeline.print_best_estimator()
    print("\tFit model runtime", time.time() - start)

    """
    ################
    Evaluating model
    ################
    """

    train_Y_hat = pipeline.predict(train_graphs)
    train_rmsd = RMSD(train_Y_hat, train_Y)

    test_Y_hat = pipeline.predict(test_graphs)
    test_rmsd = RMSD(test_Y_hat, test_Y)

    for i,elec_prop in enumerate(elec_prop_list):
        print("""
            ##########
            ### {} ###
            ##########
            """.format(elec_prop)
            )
        print("Train RMSD: {:.3f}eV".format(train_rmsd[i]))
        print("Test RMSD: {:.3f}eV".format(test_rmsd[i]))

    return list(train_rmsd) + list(test_rmsd)
