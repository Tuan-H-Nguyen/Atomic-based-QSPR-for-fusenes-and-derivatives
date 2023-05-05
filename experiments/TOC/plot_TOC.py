import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys, os
path = os.path.dirname(os.path.realpath(__file__))
lib_path = path.split("\\")
sys.path.append("\\".join(lib_path[:-1]))

import time
import numpy as np
import pickle

import matplotlib.font_manager as font_manager
from utils.plot_utility_v3 import scatter_plot
annotate = {'fontname':'Times New Roman','weight':'bold','size':20}
font_legend = font_manager.FontProperties(family = 'Times New Roman',size = 20)
custom_tick = {'fontname':'Times New Roman','size':20}
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

kernel_str = "subtree"
k = 0

train_set_sizes = []
shift = []

active_result = []
error_a = []
std_a = []

random_result = []
error_r = []
std_r = []

data = "mixed"
with open(path + "\\active_learning\\active_learning_"+kernel_str+"_"+ data+".pkl","rb") as handle:
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

elec_prop = "BG"
plot = scatter_plot()

i = 0
j = 0

X = np.array(train_set_sizes[i])*100/train_set_sizes[i][-1]
plot.add_plot(
    X, error_a[i][:,j],
    label = "Active",
    plot_line = True, 
    scatter_marker = ".",
    scatter_color = "b", line_color = "b")

plot.ax.fill_between(
    X, 
    error_a[i][:,j] - std_a[i][:,j],
    error_a[i][:,j] + std_a[i][:,j],
    #capsize = 2.0, elinewidth = 0.5, capthick = 0.5, fmt = "none",
    color = "b", alpha = 0.1
    )

plot.add_plot(
    X, error_r[i][:,j], 
    label = "Random",
    plot_line = True, 
    scatter_marker = ".",
    scatter_color = "orange", line_color = "orange",
    xticks_format = 0,
    yticks_format = 2 if i == 0 else -1,
    y_major_tick = 0.05,
    y_minor_tick = 0.01,
    xlabel = "% of full-size training set",
    ylabel = "Test RMSD for {} (eV)".format(elec_prop) if i == 0 else None,
    ylim = (0.08,0.37) if k == 0 else (0.08,0.35),
    custom_tick = custom_tick,
    custom_annotate = annotate
    )

plot.ax.fill_between(
    X,
    error_r[i][start:,j] - std_r[i][start:,j],
    error_r[i][start:,j] + std_r[i][start:,j],
    color = "orange", alpha = 0.1
    )

plot.ax.legend(
    prop = font_legend,
    )
plot.save_fig(path+"\\TOC\\active_learning_"+kernel_str+"_"+elec_prop+".jpeg",dpi =600)
fig1_path = path+"\\TOC\\active_learning_"+kernel_str+"_"+elec_prop+".jpeg"

#########################################################################################
#########################################################################################
#########################################################################################

path = os.path.dirname(os.path.realpath(__file__)).split("\\")
sys.path.append("\\".join(path[:-1]))

random_state = 315
elec_prop_list = ["BG","EA","IP"]
elec_prop_full_list = [
    "Bandgap","EA","IP"
    ]
dataset_list = ["mixed","pah","subst"]
models_list = [
    #("ecfp", "rr"),
    #("subtree", "rr"),
    #("edge", "rr"),
    ("ecfp", "gpr"),
    ("subtree", "gpr"),
    ("edge", "gpr"),
    ("shortest_path", "gpr"),
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

ylim_list = [(0.07,0.33),(0.03,0.30),(0.04,0.25)]

labels_list = ["A","B","C","D","E","F","G","H","I"]

e = 0; prop = "BG"
plot = scatter_plot()

i = 0; dataset = "mixed"

plot_data = []
mean_error = []
for m,model in enumerate(models_list):
    kernel, model = model
    test_result = np.loadtxt(
        "\\".join(path) + "\\single_run_result\\" + \
        "s_"+ dataset + "_" + kernel + "_" + model + "_" + str(random_state) + ".txt")

    test_result = test_result[:,len(elec_prop_list):]

    plot_data.append(test_result[:,e])
    mean_error.append(np.mean(test_result[:,e]))

    print(prop + " " + dataset + " " + model + " " + kernel + ": {}".format(np.mean(test_result[:,e])))

plot.add_plot(
    [i+1 for i in range(len(models_list))],mean_error,
    #ylim = ylim_list[e],
    yticks_format = 2 if i == 0 else -1,
    y_major_tick = 0.05,
    xlabel = "Models",
    ylabel = "RMSD for {}(eV)".format(elec_prop_full_list[e]) if i == 0 else None,
    scatter_marker = "s",
    scatter_color = "black",
    custom_tick = custom_tick,
    custom_annotate = annotate
    )

plot.ax.boxplot(
    plot_data,
    labels = [name_exchange(model) for model in models_list]
    )


plot.save_fig( "\\".join(path) + "\\TOC\\GPR_"+prop+".jpeg")

merge_image2(
    "\\".join(path) + "\\TOC\\GPR_"+prop+".jpeg",
    fig1_path,
    ).save("\\".join(path) + "\\TOC\\TOC.jpeg")



