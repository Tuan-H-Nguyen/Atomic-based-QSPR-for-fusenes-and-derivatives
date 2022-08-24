#%%
import copy
import numpy as np
import pandas as pd
import io
from PIL import Image
from matplotlib.pyplot import imshow, scatter
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.linear_model import LinearRegression

from data_structures.molecular_graph import MolecularGraph
from poly_rings.rings import PolyRingGraph

from utils.plot_utility import scatter_plot

from data import get_total_data, ReducedData, split_data
from ecfp import WLOrigin, generate_fpgen
from model_lr_ecfp import RR_ECFP
from mol_viewer_illustration import retrieve_smi_idx, DPO_eval, standardize, visual_longest
from mol_viewer_illustration import longest_contr_percen_dpo
from mol_viewer_illustration import find_idx_contr_atoms_of_longest


def pearson_corr(X,Y):
    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    X_m = np.mean(X)*np.ones(len(X)) - X
    Y_m = np.mean(Y)*np.ones(len(Y)) - Y
    
    R = np.sqrt(np.sum(X_m**2))*np.sqrt(np.sum(Y_m**2))
    R = np.sum(np.dot(X_m.T,Y_m)) / R
    
    return R

def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img

def assign_attribute_all_atoms(smi,model,fpgen,radius):
    model = model.best_estimator_
    fp = WLOrigin(smi)

    for _ in range(radius):
        fp.update()
        
    key2value = dict(zip(
        fpgen.unique_hash,
        list(model.coef_.squeeze())
        ))

    all_atoms = fp.graph.list_all_atoms()

    atoms = sorted(
        all_atoms, 
        key = retrieve_smi_idx)

    all_atoms_contribute = np.zeros(len(atoms))

    for label_f in fp.all_label_f:
        labels = [label_f[a] for a in atoms]
        atoms_contribute = [key2value[label] for label in labels]
        all_atoms_contribute += np.array(atoms_contribute) 
    return all_atoms_contribute

#%%
###########################################################
########### INPUT ZONE ####################################
###########################################################

radius = 3

data_type = "subst"

reload_min_max_contr = True

###########################################################

total_gen = ReducedData(
    10,10,
    pah_only=True if data_type == "pah" else False, 
    subst_only=True if data_type == "subst" else False
)

if data_type == "subst":
    save_path = "[interpretation_result2]" 
if data_type == "pah":
    save_path = "[interpretation_result]" 

total = total_gen()
total.loc[:,"smiles"] = total.loc[:,"smiles"].apply(
    standardize)

fpgen = generate_fpgen(total = total,radius = radius)

#total = get_reduced_data(10)

train, test = split_data(
    total, train_size = round(len(total)*0.75),
    random_state = 10)

best_models, test_errors = RR_ECFP(
    train,test,fpgen = fpgen,cv = 5)

bg_model = best_models[0].best_estimator_
# pah
# bg_model.intercept_ = 4.29
# np.mean(total.BG) = 3.21
# subst
# bg_model.intercept_ = 3.72
# np.mean(total.BG) = 2.56

ea_model = best_models[1].best_estimator_
# for pah
# ea_model.intercept_ = 1.50
# np.mean(total.EA) = 2.13
# subst
# ea_model.intercept_ = 3.21
# np.mean(total.EA) = 3.35

ip_model = best_models[2].best_estimator_
# for pah
# ip_model.intercept_ = 5.79
# np.mean(total.IP) = 5.35
# subst
# ip_model.intercept_ = 6.91
# np.mean(total.IP) =  5.94

X = total.loc[:,"smiles"]
Y = np.array(total.loc[:,"BG"])
X = np.array(list(map(
    fpgen.vectorize, X
)))

Y_predict = bg_model.predict(X).reshape(-1)

abs_error = abs(Y-Y_predict)

if data_type == "subst" and reload_min_max_contr:
    max_contr =  [0.13461070270710213, 0.13270167885056625, 0.07908498736478653]
    min_contr =  [-0.1645153647596928, -0.1493045299159073, -0.13768541613708152]    

elif data_type == "pah" and reload_min_max_contr:
    max_contr =  [0.1071469107910915, 0.08753959932277636, 0.051082702411423535]
    min_contr =  [-0.15496159180206073, -0.055948477401402845, -0.06743947698483828]
    
try:
    print(min_contr, max_contr)

except NameError:
    min_contr = [100]*3
    max_contr = [-100]*3
    for i,model in enumerate(best_models):
        for fp in total.smiles:
            all_contr = assign_attribute_all_atoms(fp,model,fpgen,radius)
            
            mi = min(all_contr)
            min_contr[i] = mi if mi < min_contr[i] else min_contr[i]
            
            ma = max(all_contr)
            max_contr[i] = ma if ma > max_contr[i] else max_contr[i]

    print("max_contr = ",max_contr)
    print("min_contr = ",min_contr)
#%%
for _ in range(len(total)):
    compound_idx = np.random.randint(0,len(total))
        
    print("idx",compound_idx)
    smi = list(total.loc[:,"smiles"])[compound_idx]

    save_list = []
    for j,model in enumerate(best_models):
        contributes = assign_attribute_all_atoms(
            smi,model,fpgen,radius)
        
        #min_contr = min(contributes)
        #max_contr = max(contributes)
        
        hit_ats = []
        atom_cols = {}
        
        neg_contr = []
        pos_contr = []
        
        for i,con in enumerate(contributes):
            if con <= 0:
                hit_ats.append(i)
                atom_cols[i] = (1.0,1.0-con/min_contr[j],1.0-con/min_contr[j])
                neg_contr.append(con)
            if con >= 0:
                hit_ats.append(i)
                atom_cols[i] = (1.0-con/max_contr[j],1.0-con/max_contr[j],1.0)
                pos_contr.append(con)
                
        mol = Chem.MolFromSmiles(smi)

        d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, 
            highlightAtoms=hit_ats,
            highlightAtomColors = atom_cols
            )

        d.FinishDrawing()
        imshow(show_png(d.GetDrawingText()))
        save_list.append(d)
        
        visual_longest(smi)

    for j,d in enumerate(save_list):
        d.WriteDrawingText(save_path + "/_test_vmol_"+str(compound_idx)+"_"+str(j)+".png")

    
# %%
# %%
plot = scatter_plot()

for i in np.arange(-10,11,1):
    if i < 0:
        plot.add_plot(
            0,i,
            scatter_color = [[1,1-i/-13,1-i/-13]],
        )
    elif i == 0:
        plot.add_plot(
            0,i,
            #scatter_color = [[1,1-i/-10,1-i/-10]],
            scatter_color = [[1,1,1]]
        )
    else:
        plot.add_plot(
            0,i,
            scatter_color = [[1-i/13,1-i/13,1]],
            xticks_format=-1,
            yticks_format=-1,
            x_minor_tick=100,
            x_major_tick=100,
            y_minor_tick=100,
            y_major_tick=100,

        )
plot.fig.set_size_inches(0.5,7)
plot.add_text(
    0.075,-10,
    "Most negative\n contributor",
    va = "top")
plot.add_text(
    0.075,10,
    "Most positive\n contributor",
    va = "bottom")

plot.save_fig("[interpretation_result2]/color_guide.jpeg",dpi=300)

# %%
