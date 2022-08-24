import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from utils.plot_utility_v3 import scatter_plot, font_legend, annotate

for data in ["mixed","pah","subst"]:
    num_iters = [0,1,2,3,4]

    with open("\\".join(path) + "\\vs_num_iter_"+ data+".pkl","rb") as handle:
        result = pickle.load(handle)

    std = np.std(result,axis= 0)
    result = np.mean(result,axis= 0)

    elec_prop_list = ["BG","EA","IP"]
    method_list = [
        "WL-A/RR", "WL-A/KRR",
        "WL-AB/RR", "WL-AB/KRR",
        "WL-AD/RR","WL-AD/KRR"]

    color = [
        "orange","blue",
        "crimson","cyan",
        "gray","black"]

    markers = [
        "^","D",
        "v","s",
        "o","p"]

    plots = [scatter_plot() for elec_prop in ["BG","EA","IP"]]
    notation = ["(A)","(B)","(C)"]

    shifted = [0.02,-0.01,-0.02,0.01,0.02,-0.02]

    for e,elec_prop in enumerate(elec_prop_list):
        for j,method in enumerate(method_list):
            if j > 3:
                if data == "pah":
                    n = 4
                elif data != "pah":
                    n = 3
            else:
                n = len(num_iters)

            plots[e].ax.errorbar(
                np.array(num_iters[:n])+shifted[j],
                result[e+j*len(elec_prop_list),:n],
                std[e+j*len(elec_prop_list),:n],
                color = color[j],
                elinewidth = 0.7,
                capthick = 0.7,
                capsize = 2.0,
                fmt = 'none'
                )

            plots[e].add_plot(
                num_iters[:n],
                result[e+j*len(elec_prop_list),:n],
                plot_line = True, label = method,
                scatter_color = color[j], scatter_marker = markers[j],
                line_color = color[j],
                xticks_format = 0, x_major_tick = 1,
                xlabel = "Number of iterations",
                ylabel = "RMSD for {} (eV)".format(elec_prop),
                )

            plots[e].ax.text(
                0.95,0.95,
                notation[e],
                ha='center', va='center', 
                transform=plots[e].ax.transAxes,
                **annotate
                )

    for e,elec_prop in enumerate(elec_prop_list):
        if e == 1:
            plots[e].ax.legend(
                loc = "center left",
                bbox_to_anchor=(1.04,0.5), borderaxespad=0,
                prop = font_legend,
                )
        plots[e].save_fig("\\".join(path)+"\\[result]\\vs_num_iter_"+data+"_"+elec_prop+".jpeg")









