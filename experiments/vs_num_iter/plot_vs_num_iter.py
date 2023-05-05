import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-2]))

import time
import numpy as np
import pickle

import matplotlib.font_manager as font_manager
from utils.plot_utility_v2 import scatter_plot, font_legend, annotate

font_legend = font_manager.FontProperties(family = 'Times New Roman',size = 15)

plot_lim_data = {
    "mixed":[(0.13,0.65), (0.09,0.5),(0.05,0.39)],
    "pah":[(0.08,0.65), (0.03,0.35),(0.03,0.30)],
    "subst":[(0.09,0.70), (0.05,0.6),(0.05,0.60)],
}

for d,data in enumerate(["mixed","pah","subst"]):
    with open("\\".join(path) + "\\pkl\\vs_num_iter_"+ data+".pkl","rb") as handle:
        result = pickle.load(handle)

    std = np.std(result,axis= 0)
    result = np.mean(result,axis= 0)

    with open("\\".join(path) + "\\pkl\\vs_ecfp_radius_"+ data+".pkl","rb") as handle:
        result2 = pickle.load(handle)

    std2 = np.std(result2,axis= 0)
    result2 = np.mean(result2,axis= 0)

    with open("\\".join(path) + "\\pkl\\vs_num_iter_"+ data+"_sp.pkl","rb") as handle:
        sp_result = pickle.load(handle)

    std3 = np.std(sp_result,axis= 0)
    result3 = np.mean(sp_result,axis= 0)

    std = np.concatenate(
        [
            std2,
            np.pad(std,((0,0),(0,len(std2[0]) - len(std[0]))) ),
            np.pad(std3,((0,0),(0,len(std2[0]) - len(std3[0]))) ),
        ], axis = 0)

    result = np.concatenate(
        [
            result2,
            np.pad(result,((0,0),(0,len(result2[0]) - len(result[0]))) ),
            np.pad(result3,((0,0),(0,len(result2[0]) - len(result3[0]))) ) ,
        ], axis = 0)

    elec_prop_list = ["BG","EA","IP"]
    method_list = [
        "RR/ECFP","GPR/ECFP",
        "RR/WL-A", "GPR/WL-A",
        "RR/WL-AB", "GPR/WL-AB",
        "RR/WL-AD","GPR/WL-AD"
        ]

    color = [
        "gray","black",
        "orange","blue",
        "crimson","cyan",
        "gray","black"
        ]

    markers = [
        "o","s",
        "^","D",
        "v","s",
        "o","p"]

    if d == 0:
        plots = [scatter_plot(nrows = 1, ncols = 3, figsize = (18,4.5))] + [
            scatter_plot(nrows = 1, ncols = 3, figsize = (18,4))
            for i in range(len(elec_prop_list)-1)]
    elif d == 1:
        plots = [scatter_plot(nrows = 1, ncols = 3, figsize = (18.3,4))] + [
            scatter_plot(nrows = 1, ncols = 3, figsize = (18,4))
            for i in range(len(elec_prop_list)-1)]
    else:
        plots = [
            scatter_plot(nrows = 1, ncols = 3, figsize = (18,4))
            for i in range(len(elec_prop_list))]

    notation = [
        "(A)","(B)","(C)",
        "(D)","(E)","(F)",
        "(G)","(H)","(I)"
        ]

    shifted = [0.02,-0.01,0.02,-0.01,-0.02,0.01,0.02,-0.02]

    for j,method in enumerate(method_list):
        print(method)
        num_iters = [0,1,2,3,4,5]
        if data == "mixed":
            if method == "GPR/WL-AB" : num_iters = [0,1,2]
            elif method == "GPR/WL-A" : num_iters = [0,1,2]
            elif method == "RR/WL-AD" : num_iters = [0,1,2,3]
            elif method == "GPR/WL-AD" : num_iters = [0,1,2]
        elif data == "pah":
            if method == "GPR/WL-AB" : num_iters = [0,1,2]
            if method == "GPR/WL-A" : num_iters = [0,1,2,3]
            if method == "RR/WL-AD" : num_iters = [0,1,2,3]
            if method == "GPR/WL-AD" : num_iters = [0,1,2]
        elif data == "subst":
            if method == "GPR/WL-AB" : num_iters = [0,1,2,3,4]
            if method == "GPR/WL-A" : num_iters = [0,1,2,3,4]
            if method == "RR/WL-AD" : num_iters = [0,1,2]
            if method == "GPR/WL-AD" : num_iters = [0,1,2]
        for e,elec_prop in enumerate(elec_prop_list):

            Y = result[e+j*len(elec_prop_list),:]
            X = np.arange(len(Y)) if j <= 1 else np.array(num_iters)

            if j <= 1:
                n = len(Y)
                subplot_idx = 0 
            else:
                n = len(num_iters)
                subplot_idx = j%2 + 1

            Y = Y[:n]

            plots[e].ax[subplot_idx].errorbar(
                X + shifted[j], Y,
                std[e+j*len(elec_prop_list),:n],
                color = color[j],
                elinewidth = 0.7,
                capthick = 0.7,
                capsize = 2.0,
                fmt = 'none'
                )

            plots[e].add_plot(
                X, Y, idx = subplot_idx,
                plot_line = True, label = method,
                scatter_color = color[j], scatter_marker = markers[j],
                line_color = color[j],
                xticks_format = 0, # if e == 2 else -1,
                yticks_format = 2 if subplot_idx == 0 else -1,
                x_major_tick = 1,
                ylim = plot_lim_data[data][e],
                xlabel = "Number of iterations" if d == 2 else None,
                #ylabel = "RMSD for {} (eV)".format(elec_prop),
                )

            plots[e].ax[subplot_idx].text(
                0.95,0.95,
                notation[d + subplot_idx*3],
                ha='center', va='center', 
                transform=plots[e].ax[subplot_idx].transAxes,
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
            for subplot_idx in range(3):
                plots[e].ax[subplot_idx].legend(
                    prop = font_legend,
                    loc="lower left",
                    bbox_to_anchor=(0,1.02,1,0.2),
                    mode="expand", borderaxespad=0,
                    ncol = 3
                    )
        if e == 0 and data == "pah":
            plots[e].add_plot(
                [],[],
                ylabel = "RMSD for {} (eV)".format(elec_prop),
                #xticks_format = 0, x_major_tick = 1,
                )
        plots[e].save_fig("\\".join(path)+"\\plot\\"+data+"_"+elec_prop+".jpeg")
        print("fig saved at ", "\\".join(path)+"\\plot\\vs_num_iter\\"+data+"_"+elec_prop+".jpeg")










