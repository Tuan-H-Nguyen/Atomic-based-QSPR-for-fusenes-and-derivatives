PATH_TO_PKL = ""

import pickle
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
parser.add_argument(
    "-o", "--output",
    help = "Write the result to output file, which can be used for plotting box plots of RMSDs",
    type = bool, default = False)

parser.add_argument(
    "-p", "--pkl_model",
    help = """
        Pickle model. The pickled file will be saved at the path specified by the variable 
        PATH_TO_PKL at the beginning of the file. WARNING: Large file size
    """,
    type = bool, default = False)

parser.add_argument(
    "-plot", "--parity_plot",
    help = """
        parity plot
    """,
    type = bool, default = False)

parser.add_argument(
    "-plot_label", "--parity_plot_label",
    help = """
        parity plot label 
    """,
    type = str, default = "ABC")

args = parser.parse_args()

if args.kernel == "subtree" or args.kernel == "wla":
   wl_labelling_method = WLSubtree
elif args.kernel == "edge" or args.kernel == "wlab" :
   wl_labelling_method = WLEdge
elif args.kernel == "shortest_path" or args.kernel == "wlad" :
   wl_labelling_method = WLShortestPath
else:
    assert args.kernel == "ecfp"
    wl_labelling_method = "ecfp"

data_generator = data_selector(args.data,"data",args.random_state)

print(args.num_iter)
all_rmsd = []
all_test_r2 = []

model_ensemble = []

for i in range(args.N):
    rmsd, test_r2, model = main_pipeline(
        model = args.model, 
        vectorizing_method = wl_labelling_method,
        num_iter = args.num_iter,
        data_generator = data_generator,
        train_split = args.train_split,
        random_state = args.random_state,
        return_model = True,
        parity_plot_path = "experiments\\single_run_result\\plot\\parity_plot_" \
            + args.data + "_" + args.kernel + "_" + args.model + ".jpeg"  
            if args.parity_plot else None,
        parity_plot_label = args.parity_plot_label
        )

    all_rmsd.append(rmsd)
    all_test_r2.append(test_r2)

    model_ensemble.append(model)

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

    if args.output:
        np.savetxt(
            "s_"+ args.data + "_" + args.kernel + "_" + args.model + "_" + str(args.random_state) + ".txt",
            np.array(all_rmsd))
        
        np.savetxt(
            "s_r2_"+ args.data + "_" + args.kernel + "_" + args.model + "_" + str(args.random_state) + ".txt",
            np.array(all_test_r2))

    if args.pkl_model:
        pkl_path = "model_ensemble_"+ args.data + "_" + args.kernel + "_" + args.model + ".pkl"
        with open(pkl_path,"wb") as handle:
            pickle.dump(model_ensemble,handle)





