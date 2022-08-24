#%% 
import numpy as np
from zlib import crc32
import time 
from collections import Counter
from itertools import chain

from data_structures.essential_data import VALENCE
from data_structures.molecular_graph import MolecularGraph
from graph_utilities import Dijkstra
from featurize import featurize_atom
from poly_rings.rings import PolyRingGraph

SORT_ORDER = {1: 1, 2: 2, 3: 3, 1.5: 4}

class WLbase:
    def __init__(
        self, nodes, adj):

        self.nodes = nodes
        self.label_f = {}
        self.all_label_f = []
        self.wl_vec = {}

        # fp for storing label: it's # of occurences
        for node in self.get_nodes():
            self.label_f.update(
                {node:self.hash(self.get_initial_info(node))}
            )
        self.all_label_f.append(self.label_f)
        
    def create_graph(self, smiles):
        graph = MolecularGraph()
        graph.from_smiles(smiles)
        if self.find_cycle:
            graph.find_cycles(mode = "minimal")
        return graph
        
    def get_initial_info(self,node):
        return featurize_atom(node,self.initial_connectivity_feat)
          
    def get_nodes(self):
        return self.graph.list_all_atoms()
    
    def get_adj(self,node):
        return list(node.connectivity.keys())

    def get_subtree_labels(self,root,height):
        result = []
        if height == 0:
            return [self.label_f[root]]

        result.append([self.label_f[root]])
        root_adj = self.get_adj(root)
        root_adj = sorted(root_adj,key = lambda x: self.label_f[x])

        root_adj = [
            self.get_subtree_labels(node,height = height - 1)
            for node in root_adj]

        result += root_adj
        result = chain(*result)

        return list(result)
    
    def hash(self,item):
        assert isinstance(item,list)
        """
        item = np.array(
            item,
            #dtype = np.int
            )
        return crc32(item.tobytes())
        """
        item = "".join([str(i) for i in item])
        hash_int = crc32(item.encode("utf8")) & 0xffffffff
        return hash_int

    def relabel(self):
        """
        Updating the label function/hashmap
        by computing the new labels of all atoms
        then storing them all in self.label_f
        """
        label_f = {}
        for node in self.get_nodes():

            """
            #the sorting is done purely based on 
            #magnitude of labels
            
            adj_nodes = [
                self.label_f[n] for n in 
                self.get_adj(node)]
            adj_nodes = sorted(adj_nodes)
            
            #insert main node
            adj_nodes.insert(0,self.label_f[node])
            label_f.update({node:self.hash(adj_nodes)})
            """

            label_f.update({
                node:self.hash(
                    self.get_subtree_labels(node,height = 1))
                    })

        self.label_f = label_f
        self.all_label_f.append(label_f)
    
    def update_wl_vec(self, list_of_labels):
        # updating the fp,
        # which record label: it's # of occurences

        new = dict(Counter(list_of_labels))
        for k,v in new.items():
            if k in self.wl_vec:
                self.wl_vec[k] += v
            else:
                self.wl_vec.update({k:v})

class WLSubtree(WLbase):
    def __init__(
        self,smiles, initial_connectivity_feat = True,
        find_cycle = False):

        super().__init__(smiles,initial_connectivity_feat,
            find_cycle)
        self.update_wl_vec(self.label_f.values())
        
    def generate(self,radius = 2):
        #main function for generating vector
        for _ in range(radius):
            self.relabel()
            self.update_wl_vec(self.label_f.values())

        return self.wl_vec

class WLEdge(WLbase):
    def __init__(self,smiles,initial_connectivity_feat=True):
        super().__init__(smiles,initial_connectivity_feat)

        self.edge_labels = []

    def labelling_edge(self):
        atoms = self.get_nodes()
        for i, atom in enumerate(atoms):
            adj_atoms = atom.connectivity
            for adj_atom, bond_type in adj_atoms.items():
                if atoms.index(adj_atom) < i: continue

                M = sorted([self.label_f[atom], self.label_f[adj_atom]])
                M.insert(0,bond_type)

                edge_hash = self.hash(M)
                self.edge_labels.append(edge_hash)

    def generate(self,radius = 2):
        #main function for generating vector
        self.labelling_edge()
        for _ in range(radius):
            self.relabel()
            self.labelling_edge()

        self.update_wl_vec(self.edge_labels)

        return self.wl_vec

class WLSubtreeEdge(WLEdge):
    def __init__(self,smiles,initial_connectivity_feat=True):
        super().__init__(smiles,initial_connectivity_feat)
        self.update_wl_vec(self.label_f.values())

    def generate(self,radius = 2):
        #main function for generating vector
        for _ in range(radius):
            self.relabel()
            self.labelling_edge()
            self.update_wl_vec(self.label_f.values())

        self.update_wl_vec(self.edge_labels)

        return self.wl_vec

class WLShortestPath(WLbase):
    def __init__(
        self,smiles, threshold = 0,
        initial_connectivity_feat=True
        ):

        super().__init__(smiles,initial_connectivity_feat)
        self.path_labels = []

        self.atoms = self.get_nodes()
        self.sp_algo = Dijkstra(self.atoms,edge_unity = True)

        self.longest_path = 0
        self.threshold = threshold

    def labelling_path(self,threshold = 0):
        for i, atom1 in enumerate(self.atoms):
            dist = self.sp_algo.find(atom1)
            for atom2, d in dist.items():
                if self.atoms.index(atom2) < i: 
                    continue
                if d < threshold: 
                    continue

                hash_list = sorted([
                    self.label_f[atom1],self.label_f[atom2]])

                hash_list.append(d)
                hash_list = self.hash(hash_list)
                self.path_labels.append(hash_list)

                if d > self.longest_path:
                    self.longest_path = d

    def generate(self,radius):
        #main function for generating vector

        self.labelling_path(self.threshold)

        for _ in range(radius):
            self.relabel()

            self.labelling_path(self.threshold)

        self.update_wl_vec(self.path_labels)

        return self.wl_vec
        
class FingerprintGenerator:
    def __init__(
            self,
            wl_class=None,
            radius=None,
            smiles=None,
            wl_class_args = {}
            ):
        self.wl_class = wl_class
        self.wl_class_args = wl_class_args

        self.radius = radius
        
        self.smiles2fp_dict = {}
        self.unique_hash = []
        
        if smiles != None:
            self.register(smiles)
        
    def generate_fp(self,smiles):
        fp_obj = self.wl_class(
            smiles,
            **self.wl_class_args)
        fp = fp_obj.generate(
            radius = self.radius)
        return fp

    def register(self,smiles):
        if isinstance(smiles,str):
            fp = self.generate_fp(smiles)
            
            self.smiles2fp_dict.update(
                {smiles: fp}
            )
            
            self.unique_hash += list(fp.keys())
            self.unique_hash = list(set(self.unique_hash))
            
        elif isinstance(smiles,list):
            for s in smiles:
                self.register(s)
                
        else: raise Exception("Unexpected type for SMILES:{}".format(type(smiles)))
        
    def vectorize(self,smiles):
        if isinstance(smiles,str):
            try:
                fp = self.smiles2fp_dict[smiles]
            except KeyError:
                fp = self.generate_fp(smiles)

            vec = self.unique_hash
            vec = list(map(
                lambda x: 0 if x not in fp.keys() else fp[x], 
                vec
            ))
            return vec
            
        elif isinstance(smiles,list):
            result = []
            for s in smiles:
                result.append(self.vectorize(smiles))
            return self.result
        
def generate_fpgen(
    total,radius,kernel_type,
    initial_connectivity_feat=True,
    threshold = 0
    ):

    assert kernel_type in ["subtree", "edge","subtreeEdge","shortestPath"]
    smiles = total.loc[:,"smiles"]

    start = time.time()
    fp_radius = radius
    print("Create generator for {}-{} {} ".format(
        fp_radius, "height Weisfeiler-Lehmans", kernel_type
        ))

    wl_class_args = {
        "initial_connectivity_feat":initial_connectivity_feat
        }
    if kernel_type == "subtree": wl_class = WLSubtree
    elif kernel_type == "edge": wl_class = WLEdge
    elif kernel_type == "subtreeEdge": wl_class = WLSubtreeEdge
    elif kernel_type == "shortestPath": 
        wl_class = WLShortestPath
        wl_class_args.update({"threshold":threshold})

    fpgen = FingerprintGenerator(
        wl_class = wl_class,
        radius = fp_radius,
        smiles = list(smiles),
        wl_class_args = wl_class_args
    )
    print("# unique hash ",len(fpgen.unique_hash))
    print("\n Initiate generator take:{}s".format(
        time.time() - start
    ))
    return fpgen
    

"""
smiles = "c1(ccccc1cc2cc3)cc2c4c3c5ccc6ccccc6c5cc4"

graph = MolecularGraph()
graph.from_smiles(smiles)

"""
