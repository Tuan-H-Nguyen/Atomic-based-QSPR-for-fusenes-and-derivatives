import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from utils.plot_utility_v2 import scatter_plot, font_legend, annotate

kernel_str = "subtree"

text = [["A","B","C"],["D","E","F"]]
start = 0

for k,kernel_str in enumerate(["subtree","edge"]):
    train_set_sizes = []
    shift = []

    active_result = []
    error_a = []
    std_a = []

    random_result = []
    error_r = []
    std_r = []

    for data in ["mixed","pah","subst"]:
        #with open("\\".join(path) + "\\experiments_active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
        with open("\\".join(path) + "\\active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
            result = pickle.load(handle)

        train_set_sizes.append(np.array(result["train_set_size"]))
        shift.append(np.ones((len(train_set_sizes[start:]))))

        active_result = np.squeeze(np.array(result["active"]),axis=0)
        error_a.append(np.mean(active_result,axis=0))
        std_a.append(np.std(active_result,axis= 0))

        random_result = np.squeeze(np.array(result["random"]),axis=0)
        error_r.append(np.mean(random_result,axis=0))
        std_r.append(np.std(random_result,axis=0))


    for j,elec_prop in enumerate(["BG","EA","IP"]):
        plot = scatter_plot(1,3,figsize = (16,4))
        for i in range(3):
            plot.add_plot(
                train_set_sizes[i], error_a[i][:,j],idx = i,
                label = "Actively selecting for training points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "b", line_color = "b")

            plot.ax[i].fill_between(
                train_set_sizes[i], 
                error_a[i][:,j] - std_a[i][:,j],
                error_a[i][:,j] + std_a[i][:,j],
                #capsize = 2.0, elinewidth = 0.5, capthick = 0.5, fmt = "none",
                color = "b", alpha = 0.1
                )

            plot.add_plot(
                train_set_sizes[i], error_r[i][:,j], idx = i,
                label = "Randomly selecting for training points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "orange", line_color = "orange",
                xticks_format = 0 ,#if j == 2 else -1,
                yticks_format = 3,
                xlabel = "Training set size (samples)"  if k == 1 else None,
                ylabel = "Test RMSD for {} (eV)".format(elec_prop),# if i == 0 else None,
                )

            plot.ax[i].fill_between(
                train_set_sizes[i],
                error_r[i][start:,j] - std_r[i][start:,j],
                error_r[i][start:,j] + std_r[i][start:,j],
                color = "orange", alpha = 0.1
                )

            plot.add_text2(0.95,0.95,"("+text[k][i]+")",idx=i)

        if k == 0:
            plot.ax[0].legend(
                prop = font_legend,
                loc="lower left",
                bbox_to_anchor=(0.8,1.04,1.6,0.4),
                mode="expand", borderaxespad=0,
                ncol = 2
                )
        plot.save_fig("experiments\\active_learning_"+kernel_str+"_"+elec_prop+".jpeg",dpi =600)
