from collections import Counter

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from algorithm import WL

def counter2vector(counter,label_list,max_len = None):
    vector = []
    for label in label_list:
        try:
            vector.append(counter[label])
        except KeyError:
            vector.append(0)
    if max_len != None: vector += [0]*(max_len - len(vector))
    return vector

def WLSubtreeCounter(
    nodes, adj,height, labels_list = None
    ):

    all_labels = []

    wl_labels = WL(nodes,adj)
    all_labels += wl_labels.atom_labels_list

    for i in range(height):
        wl_labels.relabelling()
        all_labels += wl_labels.atom_labels_list

    counter = dict(Counter(all_labels))

    return counter

def kernel(counter1,counter2,K):
    unique_labels = list(counter1.keys()) + list(counter2.keys())
    unique_labels = list(set(unique_labels))
    v1 = np.array([
        counter1[label] if label in counter1.keys() else 0 
        for label in unique_labels])
    v2 = np.array([
        counter2[label] if label in counter2.keys() else 0 
        for label in unique_labels])
    return K(v1,v2)

def linear_kernel(X,Y):
    """
    Input:
    X,Y is 2D matrices
    X_i, Y_i is a feature vector

    Output:
    Z is a 1D matrices
    Z_i is the rbf kernel between X_i and Y_i
    """
    #return np.einsum("ij,ij->i",X,Y)
    return np.dot(X.T,Y)
 

def rbf_kernel(X,Y,gamma):
    """
    Input:
    X,Y is 2D matrices
    X_i, Y_i is a feature vector

    Output:
    Z is a 1D matrices
    Z_i is the rbf kernel between X_i and Y_i
    """
    X = np.array(X)
    Y = np.array(Y)

    R = X - Y
    R = linear_kernel(R,R)
    return np.exp(-gamma*R)

class GraphCounter2Vector:
    def  __init__(
        self,labels,
        vector, output_dim):

        self.master_vector = vector
        self.master_labels = labels

        self.output = np.zeros(output_dim)
        self.n = 0

    def add(self,label,vector):
        
        self.output[self.n] = vector
        self.n += 1

    def __call__(self):
        return self.output

class WLSubtreeKernel(BaseEstimator):
    def __init__(self, height, kernel = "rbf", alpha = 0.1, gamma = 10e-3):
        if kernel == "rbf":
            self.kernel = lambda x,y: rbf_kernel(x,y,gamma)
        elif kernel == "linear":
            self.kernel = lambda x,y: linear_kernel(x,y)

        self.alpha = alpha

        self.height = height

    def fit(self, nodes, adj_lists, Y):
        start = time.time()

        counters_list = []
        # extracting the WL atomic labels of all training molecules
        # list of their labels , serve as indices for count of labels
        # their vector of counts of labels
        # counter of labels (labels : number of occurences)
        for i, node in enumerate(nodes):
            counter = WLSubtreeCounter(node,adj_lists[i],height=self.height)
            counters_list.append(counter)
        print(time.time() - start)

        # compute similarity btw one graph to the rest of the training set => array 
        # for that one molecule, generate its vector of labels count using 
        #    each of all graph's list of labels as indices
        # compute the similarity using previously generated vector and vectors of all graphs
        X = []
        for i,counter in enumerate(counters_list):
            x = [kernel(counter,c,self.kernel) for c in counters_list]
            X.append(x)
        X = np.array(X)

        print(time.time() - start)

        # in the feature list X, each molecules features are its similarity with the rest of the tranining set
        # ridge model is the regressor that map X to Y
        self.ridge = Ridge(self.alpha)
        self.ridge.fit(X,Y)

        print(time.time() - start)

        self.counters_list = counters_list

    def predict(self,node,adj):
        start = time.time()
        counter = WLSubtreeCounter(node,adj,height=self.height)
        
        print("Prediction time",time.time() - start)

        x = np.array([
            kernel(counter,x,self.kernel)
            for x in self.counters_list])

        print("Prediction time",time.time() - start)
        y_hat = self.ridge.predict(x.reshape(1,-1))
        print("Prediction time",time.time() - start)

        return y_hat

if __name__ == "__main__":
    import time
    import sys, os

    path = os.getcwd().split("\\")
    path = "//".join(path[:-1])

    sys.path.append(path + "//molecular_graph")

    from smiles import smiles2graph

    data = pd.read_csv(path + "//sample_DATA//raw_pah_data.csv")

    smiles = list(data.loc[:,"smiles"])
    y = np.array(data.loc[:,"BG"])

    atoms, bonds, adj, bond_feat = smiles2graph(smiles)

    model = WLSubtreeKernel(kernel = "rbf",height=2)

    start = time.time()

    model.fit(atoms,adj,y)

    print(time.time() - start)

    start = time.time()

    sample = smiles[192]
    atoms, bonds, adj, bond_feat = smiles2graph(sample)

    print(model.predict(atoms,adj))
    print(y[192])

    print("Prediction time",time.time() - start)









   
    
