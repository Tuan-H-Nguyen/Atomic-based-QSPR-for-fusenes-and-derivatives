import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__))
lib_path = path.split("\\")
sys.path.append("\\".join(lib_path[:-2]))

import time
import numpy as np
import pickle

from utils.plot_utility_v2 import scatter_plot, font_legend, annotate
from utils.Im_stitch import merge_image2, merge_image3

kernel_str = "subtree"

text = [["A","B","C"],["D","E","F"],["G","H","K"]]
start = 0

def best_index(mean_result,error=None):

    if error == None:
        return np.argmin(mean_result)
    for i,e in enumerate(mean_result):
        if e - error < 0.0001:
            break

    return i

for k,kernel_str in enumerate(["subtree","edge","shortest_path"]):
    train_set_sizes = []
    shift = []

    active_result = []
    error_a = []
    std_a = []

    random_result = []
    error_r = []
    std_r = []

    for data in ["mixed","pah","subst"]:
        print(data)
        #with open("\\".join(path) + "\\experiments_active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
        with open(path + "\\pkl\\active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
            result = pickle.load(handle)

        _train_set_sizes = np.array(result["train_set_size"])
        train_set_sizes.append(_train_set_sizes)
        shift.append(np.ones((len(train_set_sizes[start:]))))

        active_result = np.array(result["active"])
        random_result = np.array(result["random"])

        mean_active_result = np.mean(active_result,axis=0)
        error_a.append(mean_active_result)
        std_active_result = np.std(active_result,axis= 0)
        std_a.append(std_active_result)

        mean_random_result = np.mean(random_result,axis=0)
        error_r.append(mean_random_result)
        std_random_result = np.std(random_result,axis=0)
        std_r.append(std_random_result)

        """
        ir = best_index(mean_random_result)

        print("R: {:.3f}eV +/- {:.3f} at {:.2f} of the train set".format(
            np.mean(random_result,axis=0)[ir][0],
            np.std(random_result,axis=0)[ir][0],
            _train_set_sizes[ir]/_train_set_sizes[-1]
            ))

        ia = best_index(
            mean_active_result,
            round(mean_random_result[ir][0],4)
            )

        print("A: {:.3f}eV +/- {:.3f} at {:.2f} of the train set ({} less datapoints, {:.2f} train set)".format(
            np.mean(active_result,axis=0)[ia][0],
            np.std(active_result,axis=0)[ia][0],
            _train_set_sizes[ia]/_train_set_sizes[-1],
            _train_set_sizes[ir]-_train_set_sizes[ia],
            (_train_set_sizes[ir]-_train_set_sizes[ia])/_train_set_sizes[-1],
            ))

        ia = best_index(mean_active_result)

        print("A: {:.3f}eV +/- {:.3f} at {:.2f} of the train set".format(
            np.mean(active_result,axis=0)[ia][0],
            np.std(active_result,axis=0)[ia][0],
            _train_set_sizes[ia]/_train_set_sizes[-1]
            ))

        ir = best_index(
            mean_random_result,
            round(mean_active_result[ia][0],4)
            )

        print("R: {:.3f}eV +/- {:.3f} at {:.2f} of the train set ({} less datapoints, {:.2f} train set)".format(
            np.mean(random_result,axis=0)[ir][0],
            np.std(random_result,axis=0)[ir][0],
            _train_set_sizes[ir]/_train_set_sizes[-1],
            _train_set_sizes[ia]-_train_set_sizes[ir],
            (_train_set_sizes[ia]-_train_set_sizes[ir])/_train_set_sizes[-1],
            ))
        """
    for j,elec_prop in enumerate(["BG","EA","IP"]):

        ymax = np.max([np.max(error_a[i][:,j]) for i in range(3)]) + 0.05
        ymin = np.min([np.min(error_a[i][:,j]) for i in range(3)]) - 0.05

        plot = scatter_plot(1,3,figsize = (16,4))
        for i, dataset_type in enumerate(["mixed", "pah", "subst"]):
            X = np.array(train_set_sizes[i])*100/train_set_sizes[i][-1]
            plot.add_plot(
                X, error_a[i][:,j],idx = i,
                label = "Actively selecting for training points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "b", line_color = "b")

            if elec_prop == "BG":
                print(kernel_str + " " + dataset_type + " " + text[k][i])
                print(error_a[i][:,j])

            plot.ax[i].fill_between(
                X, 
                error_a[i][:,j] - std_a[i][:,j],
                error_a[i][:,j] + std_a[i][:,j],
                #capsize = 2.0, elinewidth = 0.5, capthick = 0.5, fmt = "none",
                color = "b", alpha = 0.1
                )

            plot.add_plot(
                X, error_r[i][:,j], idx = i,
                label = "Randomly selecting for training points",
                plot_line = True, 
                scatter_marker = ".",
                scatter_color = "orange", line_color = "orange",
                xticks_format = 0 if k == 2 else -1,
                yticks_format = 2 if i == 0 else -1,
                y_major_tick = 0.05,
                y_minor_tick = 0.01,
                xlabel = "% of full-size training set"  if k == 2 else None,
                ylabel = "Test RMSD for {} (eV)".format(elec_prop) if i == 0 else None,
                ylim = (ymin,ymax) 
                )

            plot.ax[i].fill_between(
                X,
                error_r[i][start:,j] - std_r[i][start:,j],
                error_r[i][start:,j] + std_r[i][start:,j],
                color = "orange", alpha = 0.1
                )

            plot.add_text2(0.95,0.95,"("+text[k][i]+")",idx=i)

        if k == 0:
            plot.ax[0].legend(
                prop = font_legend,
                loc="lower left",
                bbox_to_anchor=(0.0,1.04,3.0,0.4),
                mode="expand", borderaxespad=0,
                ncol = 2,
                markerscale = 3.0
                )
        plot.save_fig(path+"\\plot\\active_learning_"+kernel_str+"_"+elec_prop+".jpeg",dpi =600)

for j,elec_prop in enumerate(["BG","EA","IP"]):
    path = os.path.dirname(os.path.realpath(__file__))
    foo = merge_image3(*[
        path+"\\plot\\active_learning_"+kernel_str+"_"+elec_prop+".jpeg" for kernel_str in ["subtree","edge","shortest_path"]])
    foo.save(
            path+"\\plot\\active_learning_"+elec_prop+".jpeg")
