import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
from copy import deepcopy

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, TruncatedSVD

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from utils.plot_utility_v3 import scatter_plot, font_legend

from pipeline import data_splitter, graph_getter, Pipeline

random_state = 2020
num_iter = 5
data_type = "subst"

data_generator = ReducedData(
    N = 100, seed = random_state, path = "data",
    pah_only = False, subst_only = True)

data = data_generator()

data.loc[:,"smiles"] =  data.loc[:,"smiles"].apply(
    lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))

smiles_list = list(data.loc[:,"smiles"])

for idx,smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    hit_ats = []
    for i,atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() != 6:
            hit_ats.append(i)

    atom_cols = {i:(1.0,0.0,0.0) for i in hit_ats}
    d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, 
        highlightAtoms=hit_ats,
        highlightAtomColors = atom_cols
        )

    d.FinishDrawing()

    d.WriteDrawingText("\\".join(path) + "\\test\\test_{}.png".format(idx))
            
