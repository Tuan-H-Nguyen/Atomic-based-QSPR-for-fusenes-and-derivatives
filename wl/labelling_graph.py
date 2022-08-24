import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

from zlib import crc32
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from molecular_graph.smiles import smiles2graph

class WL:
    def __init__(
        self,
        nodes_feat, adj, 
        edges = None, edges_feats = None
        ):

        #initiate labels for each node by hashing a list of it properties.
        self.adj = adj
        self.atom_labels = [[
            self.hash(feat) for feat in nodes_feat]]

    def hash(self,l):
        """
        Return an integer from a list by hashing it
        """
        strl = "".join([str(i) for i in l])
        hash_int = crc32(strl.encode("utf8")) & 0xffffffff
        return hash_int

    def get_adj(self,atom_idx):
        return self.adj[atom_idx]

    def relabelling_nodes(self):
        atom_labels = self.atom_labels[-1]
        new_atomic_labels = []

        for a1,atom_label in enumerate(atom_labels):
            adj_atoms_indices = self.get_adj(a1)
            M = [
                atom_labels[idx] for idx in adj_atoms_indices]
            
            M = sorted(M)
            M.insert(atom_labels[a1],0)
            new_atomic_labels.append(
                self.hash(M))

        self.atom_labels.append(new_atomic_labels)

class WLSubtree(WL):
    def __init__(self, nodes,adj, edges=None, edges_feats=None,sp_dists=None):
        super().__init__(nodes,adj)

    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()

        atom_labels = chain(*self.atom_labels)
        return Counter(atom_labels)
        

class WLEdge(WL):
    def __init__(self,nodes,adj,edges,edges_feats,sp_dists=None):
        super().__init__(nodes,adj)

        self.edges = edges
        self.edges_feats = edges_feats

        self.edge_labels = []
        self.relabelling_edges()

    def relabelling_edges(self):
        edge_labels = []
        atom_labels = self.atom_labels[-1]

        for i,edge in enumerate(self.edges):
            a1_idx,a2_idx = edge
            M = sorted(
                [atom_labels[idx] for idx in [a1_idx,a2_idx]])
            M += self.edges_feats[i]
            edge_labels.append(self.hash(M))

        self.edge_labels.append(edge_labels)

    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()
            self.relabelling_edges()

        edge_labels = chain(*self.edge_labels)
        return Counter(edge_labels)

class WLShortestPath(WL):
    def __init__(self,nodes,adj,edges,edges_feats,sp_dists):
        super().__init__(nodes,adj)

        self.num_nodes = len(nodes)
        self.adj = adj

        self.sp_dists = sp_dists

        self.path_labels = []
        self.relabelling_path()

    def relabelling_path(self):
        path_labels = []

        atom_labels = self.atom_labels[-1]

        for i,label in enumerate(atom_labels):
            for atom_idx, path_len in enumerate(self.sp_dists[i]):
                if i > atom_idx: continue 
                M = sorted([
                    label, atom_labels[atom_idx]])
                M.append(path_len)
                path_labels.append(self.hash(M))

        self.path_labels.append(path_labels)
        
    def to_counter(self,num_iters):
        for i in range(num_iters):
            self.relabelling_nodes()
            self.relabelling_path()

        path_labels = chain(*self.path_labels)
        return Counter(path_labels)

class GraphVectorizer(BaseEstimator):
    def __init__(
        self,label_method=None,num_iter=None, smiles = True
        ):

        self.unique_labels = []
        self.num_iter = num_iter
        self.label_method = label_method

        self.smiles = smiles

    def convert_smiles(self,X):
        X = smiles2graph(X,sp=True)
        return X

    def fit(self, X,Y=None):
        if self.smiles: X=self.convert_smiles(X)

        for graph in X:
            counter = self.label_method(*graph).to_counter(self.num_iter)
            self.unique_labels += list(counter.keys())
            self.unique_labels = list(set(self.unique_labels))
        return self

    def vectorize(self,graph):
        if self.smiles: graph=self.convert_smiles(graph)

        counter = self.label_method(*graph).to_counter(self.num_iter)
        x = []
        for label in self.unique_labels:
            try: 
                x.append(counter[label])
            except KeyError: x.append(0)
        return x

    def transform(self,graphs):
        X = np.zeros((len(graphs), len(self.unique_labels)))
        for i,graph in enumerate(graphs):
            x = self.vectorize(graph)
            X[i] += np.array(x)
        return X

class GraphHashVectorizer:
    def __init__(self,vec_len, label_method, num_iter):
        self.label_method = label_method
        self.num_iter = num_iter
        self.vec_len = vec_len

    def vectorize(self,graph):
        counter = self.label_method(*graph).to_counter(self.num_iter)
        x = np.zeros(self.vec_len)
        for label,count in counter.items():
            x[label%self.vec_len] += count
        return x

    def bulk_vectorize(self,graphs):
        X = np.zeros((len(graphs), self.vec_len))
        for i,graph in enumerate(graphs):
            x = self.vectorize(graph)
            X[i] += np.array(x)
        return X

if __name__ == "__main__":
    import time
    import sys, os

    path = os.getcwd().split("\\")
    path = "//".join(path[:-1])

    sys.path.append(path)
    sys.path.append(path + "//molecular_graph")

    from smiles import smiles2graph
    data = pd.read_csv(path + "//sample_DATA//raw_cyano_data.csv")
    data = data.dropna()

    smiles = list(data.loc[:,"smiles"])
    y = np.array(data.loc[:,"BG"])

    graphs = smiles2graph(smiles)

    start = time.time()
    
    graph_vectorizer = GraphVectorizer(
        graphs,label_method = WLSubtree,num_iter = 2)

    print(time.time() - start)

    print(len(graph_vectorizer.unique_labels))

    start = time.time()

    print(smiles[1])
    print(graphs[1])
    wl_labels = WL(*graphs[1])
    wl_labels.relabelling_nodes()
    print(wl_labels.atom_labels)

    print("Featurizing everything runtime",time.time() - start)

            

            
