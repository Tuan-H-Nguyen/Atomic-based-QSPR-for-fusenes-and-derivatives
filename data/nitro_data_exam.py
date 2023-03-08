import re
import os, sys

sys.path.append("C:\\Users\\hoang\\Dropbox\\comp_chem_lab\\MLmodel")

from wl.labelling_graph import WLSubtree
from molecular_graph.smiles import smiles2graph
from molecular_graph.featurize import featurize_atom

import pandas
from zlib import crc32

from rdkit import Chem
from rdkit.Chem import Draw

def hash(l):
    """
    Return an integer from a list by hashing it
    """
    strl = "".join([str(i) for i in l])
    hash_int = crc32(strl.encode("utf8")) & 0xffffffff
    return hash_int

data = pandas.read_csv("raw_nitro_data.csv")

smiles_list = data.loc[:,"smiles"]

mol_signature = {}
for smiles in smiles_list:
    try:
        smiles = smiles.replace(r"(N(=O)=O)", '')
        smiles = smiles.replace(r"N(=O)=O", '')
    except AttributeError:
        continue

    if "N" in smiles:
        raise Exception("Nope")

    nodes_feats, adj_list, edges_list, edges_feat = smiles2graph(smiles)

    wl_graph = WLSubtree(nodes_feats, adj_list)

    flag = True
    no_unique_label = 0 
    while flag:
        wl_graph.relabelling_nodes()
        no = len(set(wl_graph.atom_labels[-1]))
        if no > no_unique_label:
            no_unique_label = no
        else: flag = False

    signature = hash(sorted(wl_graph.atom_labels[-1]))

    if signature not in mol_signature.keys():
        mol_signature.update({signature:smiles})

for sign, smiles in mol_signature.items():
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol,"nitro\\" +str(smiles.count('c')) +'_' + str(sign) + ".png")

