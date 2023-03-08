import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from utils.plot_utility_v3 import scatter_plot, font_legend, annotate

plot_lim_data = {
    "mixed":[(0.13,0.65), (0.09,0.5),(0.05,0.39)],
    "pah":[(0.08,0.65), (0.03,0.35),(0.03,0.30)],
    "subst":[(0.09,0.70), (0.05,0.6),(0.05,0.60)],
}

for d,data in enumerate(["mixed","pah","subst"]):
    with open("\\".join(path) + "\\vs_num_iter_"+ data+".pkl","rb") as handle:
        result = pickle.load(handle)

    std = np.std(result,axis= 0)
    result = np.mean(result,axis= 0)

    with open("\\".join(path) + "\\vs_num_iter_"+ data+"_sp.pkl","rb") as handle:
        sp_result = pickle.load(handle)

    std = np.concatenate(
        [
            std,
            np.std(sp_result,axis= 0),
        ])

    result = np.concatenate(
        [
            result,
            np.mean(sp_result,axis= 0)
        ])

    elec_prop_list = ["BG","EA","IP"]
    method_list = [
        "WL-A/RR", "WL-A/GPR",
        "WL-AB/RR", "WL-AB/GPR",
        "WL-AD/RR","WL-AD/GPR"
        ]

    color = [
        "orange","blue",
        "crimson","cyan",
        "gray","black"
        ]

    markers = [
        "^","D",
        "v","s",
        "o","p"]

    plots = [scatter_plot() for elec_prop in ["BG","EA","IP"]]
    notation = ["(D)","(E)","(F)"]

    shifted = [0.02,-0.01,-0.02,0.01,0.02,-0.02]

    for j,method in enumerate(method_list):
        if data == "mixed":
            num_iters = [0,1,2,3,4,5]
            if method == "WL-AB/GPR" : num_iters = [0,1,2]
            if method == "WL-A/GPR" : num_iters = [0,1,2,3]
            if method == "WL-AD/RR" : num_iters = [0,1,2,3]
            if method == "WL-AD/GPR" : num_iters = [0,1,2]
        elif data == "pah":
            num_iters = [0,1,2,3,4,5]
            if method == "WL-AB/GPR" : num_iters = [0,1,2]
            if method == "WL-A/GPR" : num_iters = [0,1,2,3]
            if method == "WL-AD/RR" : num_iters = [0,1,2,3]
            if method == "WL-AD/GPR" : num_iters = [0,1,2]
        elif data == "subst":
            num_iters = [0,1,2,3,4,5]
            if method == "WL-AB/GPR" : num_iters = [0,1,2]
            if method == "WL-A/GPR" : num_iters = [0,1,2,3]
            if method == "WL-AD/RR" : num_iters = [0,1,2]
            if method == "WL-AD/GPR" : num_iters = [0,1,2]
        for e,elec_prop in enumerate(elec_prop_list):
            if j > 3:
                pass
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
                xticks_format = 0, # if e == 2 else -1,
                yticks_format = -1, # if e == 2 else -1,
                x_major_tick = 1,
                ylim = plot_lim_data[data][e],
                xlabel = "Number of iterations" if d == 2 else None,
                #ylabel = "RMSD for {} (eV)".format(elec_prop),
                )

            plots[e].ax.text(
                0.95,0.95,
                notation[d],
                ha='center', va='center', 
                transform=plots[e].ax.transAxes,
                **annotate
                )

    for e,elec_prop in enumerate(elec_prop_list):
        if e == 0 and data == "mixed":
            """
            plots[e].add_plot(
                [],[],scatter_color = "gray", scatter_marker = "o",
                label = "ECFP/RR"
                )
            plots[e].add_plot(
                [],[],scatter_color = "black", scatter_marker = "p",
                label = "ECFP/GPR",
                #xlabel = "Number of iterations",
                ylabel = "RMSD for {} (eV)".format(elec_prop),
                xticks_format = 0, x_major_tick = 1,
                )
            """
            plots[e].ax.legend(
                prop = font_legend,
                loc="lower left",
                bbox_to_anchor=(0,1.02,1,0.2),
                mode="expand", borderaxespad=0,
                ncol = 3
                )
        plots[e].save_fig("\\".join(path)+"\\[result]\\vs_num_iter_"+data+"_"+elec_prop+".jpeg")
        print("fig saved at ", "\\".join(path)+"\\[result]\\vs_num_iter_"+data+"_"+elec_prop+".jpeg")










