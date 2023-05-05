PATH_TO_PKL = ""

import argparse
import pickle

import numpy as np

from molecular_graph.smiles import smiles2graph

parser = argparse.ArgumentParser()

parser.add_argument(
    "smiles", help = "SMILES string of molecule whose electronic properties will be predicted")

parser.add_argument(
    "-m","--model", 
    help = """
    Regression model to use. rr for ridge regression, krr for Kernel Ridge with RBF kernel,
    krr_poly for Kernel Ridge with polynomial kernel and krr_linear for linear kernel. Default to krr
    """, 
    type = str, choices = ["rr","krr","krr_poly","krr_linear","gpr","gpr_","linear"],
    default = "krr")
parser.add_argument(
    "-k","--kernel",
    help = "WL kernel type. Choice: subtree/wla, edge/wlab, shortest_path/wlad.\
    Alternatively, choose ecfp for ECFP fingerprint. Default is subtree/wla",
    type = str, choices = ["wla","subtree", "edge", "wlab", "shortest_path", "wlad", "ecfp"],
    default = "subtree")
parser.add_argument(
    "-d","--data", 
    help = "Data of choice. mixed, pah, and subst for substituted PAH. Default to mixed.",
    type = str, choices = ["mixed","pah","subst"], default = "mixed"
    )

args = parser.parse_args()

if args.kernel == "subtree" or args.kernel == "wla":
    kernel = "subtree"
elif args.kernel == "edge" or args.kernel == "wlab" :
    kernel = "edge"
elif args.kernel == "shortest_path" or args.kernel == "wlad" :
    kernel = "shortest_path"

pkl_path = PATH_TO_PKL + "model_ensemble_"+ args.data + "_" + kernel + "_" \
    + args.model + ".pkl"

with open(pkl_path,"rb") as handle:
    model_ensemble = pickle.load(handle)

graph = smiles2graph([args.smiles])

predictions = []
for model in model_ensemble:
    predictions.append(model.predict(graph))

predictions = np.mean(predictions, axis = 0)[0]

print("Predictions for Bandgap(eV): {:.2f}".format(predictions[0]))
print("Predictions for Electron Affinity(eV): {:.2f}".format(predictions[1]))
print("Predictions for Ionization Potential(eV): {:.2f}".format(predictions[2]))









