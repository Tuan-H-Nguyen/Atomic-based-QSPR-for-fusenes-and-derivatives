import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
#sys.path.append("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-2]))

import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utility_v2 import scatter_plot

random_state = 2022
elec_prop_list = ["BG","EA","IP"]
elec_prop_full_list = [
    "Bandgap","EA","IP"
    ]
dataset_list = ["mixed","pah","subst"]
models_list = [
    ("ecfp", "rr"),
    ("subtree", "rr"),
    ("edge", "rr"),
    ("shortest_path", "rr"),
    #("ecfp", "gpr"),
    #("subtree", "gpr"),
    #("edge", "gpr"),
    #("shortest_path", "gpr"),
]

def name_exchange(model):
    kernel, model = model
    kernel_names = {
        "subtree":"WL-A", "edge":"WL-AB", "shortest_path":"WL-AD",
        "ecfp":"ECFP"
        }
    kernel = kernel_names[kernel]
    model = model.upper()
    return model + "/\n" + kernel

ylim_list = [(0.07,0.35),(0.03,0.35),(0.03,0.30)]

labels_list = ["A","B","C","D","E","F","G","H","I"]

for e,prop in enumerate(elec_prop_list):
    plot = scatter_plot(1,3,figsize = (18,4))
    for i, dataset in enumerate(dataset_list):
        plot_data = []
        mean_error = []
        for model in models_list:
            kernel, model = model
            test_result = np.loadtxt(
                "\\".join(path) + "\\np_txt\\" + \
                "s_r2_"+ dataset + "_" + kernel + "_" + model + "_" + str(random_state) + ".txt")

            test_result = test_result[:,:]
            plot_data.append(test_result[:,e])
            mean_error.append(np.mean(test_result[:,e]))

            print(prop + " " + dataset + " " + model + " " + kernel + ": {}".format(np.mean(test_result[:,e])))

        plot.add_plot(
            [i+1 for i in range(len(models_list))],mean_error,idx = i,
            #ylim = ylim_list[e],
            #yticks_format = 3 if i == 0 else -1,
            xlabel = "Models" if e == 2 else None,
            ylabel = "R-squared for {}".format(elec_prop_full_list[e]) if i == 0 else None,
            scatter_marker = "s",
            scatter_color = "black"
            )

        plot.ax[i].boxplot(
            plot_data,
            labels = [name_exchange(model) for model in models_list]
            )

        if e != 2:
            plot.add_plot(
                [],[],idx = i,
                xticks_format = -1,
                yticks_format = 2,# if i == 0 else -1,
                xlabel = "Models" if e == 2 else None,
                ylabel = "R-squared for {}".format(elec_prop_full_list[e]) if i == 0 else None,
                )

        plot.add_text(
            0.95,0.10, text = "(" + labels_list[i+e*3] + ")",
            va = "top", ha = "right",idx = i)

    plot.save_fig( 
        "\\".join(path) + "\\" + \
        "plot\\RR_r2_"+prop+".jpeg")




