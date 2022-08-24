"""
Original code at:
https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py
"""
import time
import numpy as np
from rdkit import Chem

from .featurize import featurize_atom, featurize_bond
from .graph_utilities import Dijkstra

def smiles2graph(
    smiles, sp = False, minimal = False
    ):
    if isinstance(smiles,list):
        graphs = []

        for s in smiles:
            graph = smiles2graph(s,sp = sp)
            graphs.append(graph)

        return graphs
            
    elif isinstance(smiles,str):
        mol = Chem.MolFromSmiles(smiles)

        node_feat  = []

        for atom in mol.GetAtoms():
            node_feat.append(featurize_atom(atom,minimal = minimal))
            
        edges_list = []
        adj_list = [[] for node in node_feat]
        edges_feat = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx() 
            a2 = bond.GetEndAtomIdx() 

            edges_list.append((a1,a2))

            adj_list[a1].append(a2)
            adj_list[a2].append(a1)

            edges_feat.append(featurize_bond(bond))

        if sp:
            sp_dists = []
            num_nodes = len(node_feat)
            sp_algo = Dijkstra(num_nodes,adj_list)

            for i in range(num_nodes):
                sp_dists.append(sp_algo.find(i))

            return node_feat, adj_list,edges_list, edges_feat, sp_dists

        else:
            return node_feat, adj_list,edges_list, edges_feat


if __name__ == "__main__":
    smiles = "c1cc2c(cc1)c1c(ccc(c1)C#N)c1c2cccc1"

    start = time.time()
    nodes, edges, adj, edges_feat, sp_dists = smiles2graph(smiles,sp = True)
    print(time.time() - start)

