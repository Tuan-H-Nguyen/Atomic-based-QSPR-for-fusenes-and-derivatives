import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
from copy import deepcopy
import pickle
import itertools, collections
from tqdm import tqdm

import time
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from molecular_graph.smiles import smiles2graph
from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import data_splitter, graph_getter, Pipeline, RMSD, model_getter

random_state = 2020

try:
    with open("experiments\\LOO_error.pkl","rb") as handle:
        loo_error = pickle.load(handle)
except FileNotFoundError:
    data_generator = ReducedData(
        N = 100, seed = random_state, path = "data",
        pah_only = False, subst_only = False)

    elec_prop = ["BG"]
    data = data_generator()

    all_smiles = list(data.loc[:,"smiles"])
    all_graphs = []
    for smiles in all_smiles:
        all_graphs.append(
            smiles2graph(smiles))
    all_Y = np.array(data.loc[:,elec_prop])

    regressor,r_param_grid = model_getter("gpr")
    model = Pipeline(
        vectorizing_method = WLSubtree, 
        gv_param_grid = {"num_iter":[2,3]},
        regressor = regressor,
        r_param_grid = r_param_grid,
        )

    LOO_error = []
    LOO_std = []

    for i in tqdm(range(len(data))):
        train_X = all_graphs[:i] + all_graphs[i+1:]
        train_Y = np.concatenate(
            [all_Y[:i,:], all_Y[i+1:,:]])

        model.fit(train_X,train_Y)

        test_X = [all_graphs[i]]
        test_Y_hat, test_std = model.predict(test_X,return_std=True)

        rmsd = np.sqrt(
            (test_Y_hat[0][0] - all_Y[i])**2
            )
        LOO_error.append(rmsd)
        LOO_std.append(test_std)

    with open("LOO_error.pkl","wb") as handle:
        pickle.dump({
            "smiles_list":all_smiles,
            "LOO_error":LOO_error
            }, handle)

errors = np.array(loo_error["LOO_error"])
smiles = loo_error["smiles_list"]
indices = list(range(len(errors)))

indices = sorted(
    indices,
    key = lambda x: errors[x],
    reverse = True)

print("mean leave-one-out error",np.mean(errors))
print("std leave-one-out error",np.std(errors))

mols = []
sub_errors  = []
for i in [1,5,8,9,10,11,16,17]:
    idx = indices[i]
    mols.append(Chem.MolFromSmiles(smiles[idx]))
    sub_errors.append(errors[idx])

img = Draw.MolsToGridImage(
    mols,molsPerRow=4,
    subImgSize=(400,200),
    legends = ["{} . LOO_BG = {:.2f}eV".format(i+1,e[0]) for i,e in enumerate(sub_errors)]
    )
img.save("experiments\\most_error.png")
