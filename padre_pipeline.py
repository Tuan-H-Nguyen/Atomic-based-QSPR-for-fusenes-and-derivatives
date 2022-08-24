import time
import numpy as np

from data.padre import padre_features, padre_labels, padre_train, padre_predict
from utils.criterion import RMSD

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from data.data import ReducedData, stratified_sampling
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import Pipeline, data_splitter, data_selector, graph_getter

def padre_model_getter(model,kernel = None,grid_search = False,name = None):
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
                "alpha" : [5e-3,5e-2],
                "_max_iter":[20000]
                }
    else:
        raise Exception("Model {} has not been implemented".format(model))

    if grid_search:
        return GridSearchCV(regressor,param_grid=param_grid)
    else:
        return regressor, param_grid

class Padre_Pipeline(Pipeline):
    def __init__(
        self,
        vectorizing_method, gv_param_grid,
        regressor, r_param_grid,
        input_smiles = False
        ):

        super().__init__(
            vectorizing_method, gv_param_grid,
            regressor, r_param_grid,
            input_smiles = False)
    
    def fit_model(self,model,X,Y):
        model = padre_train(model,X,Y)
        return model

    def labels(self,Y):
        #Y = Y.reshape(-1)
        return Y

    def predict(self,X):
        for i in range(self.n_prop):
            X_ = self.graph_vectorizers[i].transform(X)
            y,y_std = padre_predict(
                self.regressors[i],
                X_, self.X, self.Y
                )
            Y.append(y)
            Y_std.append(y_std)
        return np.array(Y).T, np.array(Y_std).T

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
    #if len(elec_prop_list) == 1:
        #train_Y, test_Y = train_Y.T, test_Y.T

    """
    ##################
    Building regressor
    ##################
    """
    start = time.time()

    regressor,param_grid = padre_model_getter(model,vectorizing_method ,False)

    pipeline = Padre_Pipeline(
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

"""
main_pipeline(
    model = "krr",
    data_generator = data_selector("mixed","data",2022),
    vectorizing_method = WLSubtree,
    num_iter = [2,3],
    train_split = 0.5,
    random_state = 2022,
    elec_prop_list = ["BG"]
    )
"""
print("Yamete kudasai! DON'T DO THAT TO YOUR COMPUTER LOL")





        
