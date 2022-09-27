import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
import pickle

import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

data_type = "subst"

def ranking(test,number_top,reverse = False):
    rank = []
    idx_list = list(range(len(test)))
    idx_list = sorted(
        idx_list,
        key = lambda i: test[i],
        reverse = reverse
        )
    return idx_list[:number_top]

with open("\\".join(path) + "\\"+data_type+"\\result.pkl","rb") as handle:
    result_dict = pickle.load(handle)

for random_state, result in result_dict.items():
    contributions_list = result["contributions_list"]
    test_rmsd = result["test_rmsd"]
    test_smiles = result["test_smiles"]
    test_Y = result["test_Y"]

    min_contr = min([min(con) for con in contributions_list])
    max_contr = max([max(con) for con in contributions_list])

    ranks = [
        ranking(test_Y,number_top = 10, reverse = True),
        ranking(test_Y,number_top = 10, reverse = False)
        ]
    for c,cat in enumerate(["high_bg","low_bg"]):
        for idx in ranks[c]:
            hit_ats = []
            atom_cols = {}
            sample = test_smiles[idx]
            contributions = contributions_list[idx]
            
            for i,con in enumerate(contributions):
                if con <= 0:
                    hit_ats.append(i)
                    atom_cols[i] = (1.0,1.0-con/min_contr,1.0-con/min_contr)
                if con >= 0:
                    hit_ats.append(i)
                    atom_cols[i] = (1.0-con/max_contr,1.0-con/max_contr,1.0)
                    
            mol = Chem.MolFromSmiles(sample)

            d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs
            """
            
            """
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, 
                highlightAtoms=hit_ats,
                highlightAtomColors = atom_cols
                )

            d.FinishDrawing()

            d.WriteDrawingText("\\".join(path) + "\\" + data_type + "\\" +"\\"+cat+"\\test_{}_{}.png".format(idx,random_state))
