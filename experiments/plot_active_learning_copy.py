import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from utils.plot_utility_v3 import scatter_plot, font_legend, annotate

data = "subst"
kernel_str = "edge"
for ks,kernel_str in enumerate(["subtree","edge"]):
    for data in ["mixed","pah","subst"]:
        with open("\\".join(path) + "\\experiments_active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
        #with open("\\".join(path) + "\\active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
            result = pickle.load(handle)

        start = 0
        if data == "subst":
            if kernel_str == "edge":
                #start = 15 #108
                start = 0 #108
        elif data == "mixed":
            if kernel_str == "edge":
                #start = 32 #261
                start = 0 #32
        elif data == "pah":
            if kernel_str == "edge":
                #start = 14 #396
                start = 0 #261

        train_set_sizes = np.array(result["train_set_size"])
        shift = np.ones((len(train_set_sizes[start:])))

        active_result = np.squeeze(np.array(result["active"]),axis=0)
        error_a = np.mean(active_result,axis=0)
        std_a = np.std(active_result,axis= 0)

        random_result = np.squeeze(np.array(result["random"]),axis=0)
        error_r = np.mean(random_result,axis=0)
        std_r = np.std(random_result,axis=0)

        for i, elec_prop in enumerate(["Bandgap","EA","IP"]):
            """
            if data == "subst":
                plot = scatter_plot(mini_plot = True,mini_plot_rect = [0.30,0.30,0.55,0.55])
            elif data == "mixed":
                plot = scatter_plot(mini_plot = True,mini_plot_rect = [0.35,0.30,0.50,0.55])
            else:
            """
            plot = scatter_plot()

            plot.add_plot(
                train_set_sizes[start:], error_a[start:,i],
                label = "Actively selecting for \ntraining points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "b", line_color = "b"
                )
            plot.ax.fill_between(
                train_set_sizes[start:], 
                error_a[start:,i] - std_a[start:,i],
                error_a[start:,i] + std_a[start:,i],
                #capsize = 2.0, elinewidth = 0.5, capthick = 0.5, fmt = "none",
                color = "b", alpha = 0.1
                )

            plot.add_plot(
                train_set_sizes[start:], error_r[start:,i],
                label = "Randomly selecting for \ntraining points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "orange", line_color = "orange",
                xticks_format = 0 if i == 2 else -1,
                xlabel = "Training set size (samples)" if i == 2 else None,
                ylabel = "Test RMSD for {} (eV)".format(elec_prop) if ks == 0 else None,
                )

            plot.ax.fill_between(
                train_set_sizes[start:],
                error_r[start:,i] - std_r[start:,i],
                error_r[start:,i] + std_r[start:,i],
                color = "orange", alpha = 0.1
                )

            if i == 0:
                plot.ax.legend(
                    prop = font_legend,
                    loc="lower left",
                    bbox_to_anchor=(0,1.02,1,0.2),
                    mode="expand", borderaxespad=0,
                    ncol = 2
                    )

            plot.save_fig("experiments\\active_learning_"+kernel_str+"_"+data+"_"+elec_prop+".jpeg",dpi =600)
