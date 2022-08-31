import argparse
import numpy as np
import pandas as pd

from data.data import ReducedData, stratified_sampling
from wl.labelling_graph import (WLSubtree, WLEdge, WLShortestPath, 
    GraphVectorizer, GraphHashVectorizer)

from pipeline import main_pipeline, data_selector

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n","--N", help = "Number of repeating experiments",
     default = 1, type = int, choices = [i for i in range(1,21,1)])

parser.add_argument(
    "-m","--model", 
    help = """
    Regression model to use. rr for ridge regression, krr for Kernel Ridge with RBF kernel,
    krr_poly for Kernel Ridge with polynomial kernel and krr_linear for linear kernel. Default to krr
    """, 
    type = str, choices = ["rr","krr","krr_poly","krr_linear","gpr"],
    default = "krr")
parser.add_argument(
    "-k","--kernel",
    help = "WL kernel type. Choice: subtree, edge, shortest_path. Alternatively, choose ecfp for ECFP fingerprint",
    type = str, choices = ["subtree", "edge", "shortest_path","ecfp"],
    default = "subtree")
parser.add_argument(
    "-d","--data", 
    help = "Data of choice. mixed, pah, and subst for substituted PAH. Default to mixed.",
    type = str, choices = ["mixed","pah","subst"], default = "mixed"
    )
parser.add_argument(
    "-i","--num_iter", nargs = "+",
    help = "The number of iteration of the WL labelling algorithm. Int < 10, default to 2.",
    type = int, default = 2, choices = range(0,20))
parser.add_argument(
    "-t","--train_split",
    help = "ratio of training set/total data set. float < 1.0, default to 0.7.",
    default = 0.7, type = float)
parser.add_argument(
    "-s", "--random_state",
    help = "Random seed", type = int, default = 2022)

args = parser.parse_args()

if args.kernel == "subtree":
   wl_labelling_method = WLSubtree
elif args.kernel == "edge":
   wl_labelling_method = WLEdge
elif args.kernel == "shortest_path":
   wl_labelling_method = WLShortestPath
else:
    assert args.kernel == "ecfp"
    wl_labelling_method = "ecfp"

data_generator = data_selector(args.data,"data",args.random_state)

print(args.num_iter)
all_rmsd = []
for i in range(args.N):
    rmsd = main_pipeline(
        model = args.model, 
        vectorizing_method = wl_labelling_method,
        num_iter = args.num_iter,
        data_generator = data_generator,
        train_split = args.train_split,
        random_state = args.random_state
        )

    all_rmsd.append(rmsd)

elec_prop_list = ["BG","EA","IP"]
avg_rmsd = np.mean(all_rmsd,axis= 0)
std_rmsd = np.std(all_rmsd,axis= 0)
for i,error in enumerate(["Train error", "Test error"]):
    print("########### " + error + " ##########")

    for e,elec_prop in enumerate(elec_prop_list):
        print("\tError on "+elec_prop + ": {:.2f}(eV)".format(
            avg_rmsd[e + i*len(elec_prop_list)]))
        print("\tSTD on "+elec_prop + ": {:.2f}(eV)".format(
            std_rmsd[e + i*len(elec_prop_list)]))







