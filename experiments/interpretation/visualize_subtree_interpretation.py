import os, sys
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))
import pickle
from math import floor, ceil

import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

import PIL
from PIL import Image, ImageDraw, ImageFont
from utils.Im_stitch import side_merge_any, merge_image3

data_type = "subst"

def ranking(test,number_top,true_idx = None,reverse = False, middle = False):
    rank = []
    idx_list = list(range(len(test)))
    idx_list = sorted(
        idx_list,
        key = lambda i: test[i],
        reverse = reverse
        )
    if not middle:
        result = idx_list[:number_top]

    elif middle:
        upper = int(len(idx_list)/2 + ceil(number_top/2))
        lower = int(len(idx_list)/2 - floor(number_top/2))
        result = idx_list[lower:upper]

    if idx_list:
        result = [true_idx[i] for i in result]

    return result

with open("\\".join(path) + "\\"+data_type+"\\result.pkl","rb") as handle:
    result_dict = pickle.load(handle)

contributions_list = result_dict["contributions_list"]

test_smiles = result_dict["test_smiles"]
test_Y = result_dict["test_Y"]

test_Y1 = []; test_Y2 = []
test_idx1 = []; test_idx2 = []

if data_type == "pah":
    for i,smiles in enumerate(test_smiles):
        if "S" in smiles or "s" in smiles:
            test_Y1.append(test_Y[i])        
            test_idx1.append(i)
        else:
            test_Y2.append(test_Y[i])        
            test_idx2.append(i)

elif data_type == "subst":
    for i,smiles in enumerate(test_smiles):
        if "O" in smiles or "o" in smiles:
            test_Y1.append(test_Y[i])        
            test_idx1.append(i)
        else:
            test_Y2.append(test_Y[i])        
            test_idx2.append(i)

min_contr = min([min(con) for con in contributions_list])
max_contr = max([max(con) for con in contributions_list])

ranks = [
    ranking(test_Y1,2,test_idx1, reverse = True) + ranking(test_Y2,2,test_idx2, reverse = True),
    ranking(test_Y1,2,test_idx1, middle=True) + ranking(test_Y2,2,test_idx2, middle=True),
    ranking(test_Y1,2,test_idx1, reverse = False) + ranking(test_Y2,2,test_idx2, reverse = False),
    ]

big_img_paths = []
for c,cat in enumerate(["high_bg","mid_bg","low_bg"]):
    img_paths = [] 
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
        
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, 
            highlightAtoms=hit_ats,
            highlightAtomColors = atom_cols
            )

        d.FinishDrawing()

        d.WriteDrawingText("\\".join(path) + "\\" + data_type + "\\"+cat+"\\test_{}.png".format(idx))

        image = Image.open("\\".join(path) + "\\" + data_type +"\\"+cat+"\\test_{}.png".format(idx))

        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(
            r'C:\Users\hoang\Dropbox\comp_chem_lab\MLmodel\interpretation\\times new roman.ttf',
            size = 50) 

        draw.text(
            (100,430),
            "BG = {:.2f}eV".format(test_Y[idx][0]),
            font=font,align="center",fill="black")

        image.save("\\".join(path) + "\\" + data_type  +"\\"+cat+"\\test_{}_.jpeg".format(idx))

        img_paths.append("\\".join(path) + "\\" + data_type  +"\\"+cat+"\\test_{}_.jpeg".format(idx))

        os.remove("\\".join(path) + "\\" + data_type + "\\"+cat+"\\test_{}.png".format(idx))

    side_merge_any(img_paths).save("\\".join(path) + "\\" + data_type+"\\test_{}_.jpeg".format(cat))
    for path_ in img_paths: os.remove(path_)
    big_img_paths.append("\\".join(path) + "\\" + data_type+"\\test_{}_.jpeg".format(cat))

color_guide = Image.open("\\".join(path) + "\\" + data_type+"\\color_guide.jpeg")
size = [0.8*dim for dim in color_guide.size]
color_guide.thumbnail(size,Image.Resampling.LANCZOS)

side_merge_any(
    [
        merge_image3(*big_img_paths),
        color_guide
        ],
    open_img = False
    ).save("\\".join(path) + "\\" + data_type+"\\final.jpeg")
for path_ in big_img_paths: os.remove(path_)

