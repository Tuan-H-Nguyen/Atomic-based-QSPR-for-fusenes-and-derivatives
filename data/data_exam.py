import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import ReducedData

import sys, os
path = os.path.dirname(os.path.realpath(__file__))
lib_path = path.split("\\")
data_path = "\\".join(lib_path[:-1])

annotate = {'fontname':'Times New Roman','weight':'bold','size':13}

total_data = ReducedData(1,1, path = data_path + "\\sample_DATA")
total_data = total_data()

def cn_filter(smiles_list):
    result = {}
    for smiles in smiles_list:
        if not "N" in smiles or "O" in smiles:
            continue
        try:
            result[smiles.count("N")] += 1
        except KeyError:
            result.update({smiles.count("N"):1})
    return result

def no2_filter(smiles_list):
    result = {}
    for smiles in smiles_list:
        if not "N" in smiles or "O" not in smiles:
            continue
        try:
            result[smiles.count("O")/2] += 1
        except KeyError:
            result.update({smiles.count("O")/2:1})
    return result

def s_filter(smiles_list):
    result = 0
    for smiles in smiles_list:
        if "S" in smiles or  "s" in smiles:
            result += 1
    return result

def s_filter2(smiles_list):
    result1 = 0
    result2 = 0
    for smiles in smiles_list:
        if "S" in smiles or  "s" in smiles:
            if smiles.count("s") == 1:
                result1 += 1
            elif smiles.count("s") == 2:
                result2 += 1
    return result1, result2

def no_rings(smiles):
    if isinstance(smiles,list):
        result = []
        for s in smiles:
            result.append(no_rings(s))
        return result
    else:
        no_c = smiles.count('c')
        no_s = smiles.count('s')
        no_c += 2*no_s
        result = (no_c - 2)/4
        return result
    

smiles_list = list(total_data.loc[:,"smiles"])

cn_dist = cn_filter(smiles_list)

no2_dist = no2_filter(smiles_list)

num_s = s_filter(smiles_list)

list_of_no_rings = no_rings(smiles_list)

"""
plt.hist(list_of_no_rings, bins = [3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5])

plt.title("A. Histogram of distribution of number of rings of the dataset.")

plt.xlabel("Number of rings (rings)")

plt.ylabel("Number of samples (samples)")

plt.savefig("data_hist.jpeg",dpi=600)

"""

"""
Initializing and setting subplots
"""
# fig size = 32
# y_gap = 2
# x_gap = 20
# grid size = 72
fig = plt.figure(figsize = (12,11))

gs = gridspec.GridSpec(66,84)

#fig,ax = plt.subplots(2,2,figsize = (12,8))

#ax[0][1].axis("off")


"""
### Pie chart for all
"""
no2 = sum(no2_dist.values())
cn = sum(cn_dist.values())
total = len(total_data)
sizes = [v for v in no2_dist.values()]

sizes += [v for v in cn_dist.values()]

sizes += [
    num_s,
    (total-cn-no2-num_s)]
explode_list = []

labels = ["{:.0f}-NO$_2$-group(s) PAHs".format(k) for k in no2_dist.keys()]
n = len(labels)
explode_list += [0.04*i for i in range(n)]

labels +=["{:.0f}-CN-group(s) PAHs".format(k) for k in cn_dist.keys()]
explode_list += [0.04*i for i in range(len(labels)-n)]
n = len(labels)

labels += ["Thienoacenes", "PAH"]
explode_list += [0.0,0.04]

ax = fig.add_subplot(gs[:32,26:58])
ax.pie(sizes[::-1],explode = explode_list[::-1],labels = labels[::-1] ,autopct="%1.f%%")

print(total)
ax.set_title("A. Distribution of the mixed dataset", 
    bbox = dict(edgecolor = "black", facecolor = "none" ),
    **annotate)

#fig.savefig(path + "\\plot\\data_pie_all.jpeg",bbox_inches = "tight",dpi=600)

"""
### Pie chart for the distribution of pah
"""
total_data = ReducedData(1,1,subst_only = False, pah_only = True, path = data_path + "\\sample_DATA")
total_data = total_data()

smiles_list = list(total_data.loc[:,"smiles"])

num_s1, num_s2 = s_filter2(smiles_list)
total = len(total_data)
print(total)

sizes = [
    num_s1,
    num_s2,
    (total-num_s1-num_s2)]

labels = ["1-thiophene\n thienoacenes","2-thiophene\n Thienoacenes", "PAH"]

#fig,ax = plt.subplots()

ax = fig.add_subplot(gs[34:, :32])
ax.pie(sizes,labels = labels,autopct="%1.1f%%")

ax.set_title("B. Distribution of the PAH dataset",
    bbox = dict(edgecolor = "black", facecolor = "none" ),
     **annotate)
#fig.savefig(path + "\\plot\\data_pie_pah.jpeg",bbox_inches = "tight",dpi=600)

"""
### Pie chart for the distribution of substituent
"""

total_data = ReducedData(1,1,subst_only = True, path = data_path + "\\sample_DATA")
total_data = total_data()

smiles_list = list(total_data.loc[:,"smiles"])

cn_dist = cn_filter(smiles_list)

no2_dist = no2_filter(smiles_list)

num_s = s_filter(smiles_list)

print(len(total_data))

sizes = [v for v in no2_dist.values()]

sizes += [v for v in cn_dist.values()]

labels = ["{:.0f}-NO$_2$-group(s) PAHs".format(k) for k in no2_dist.keys()]
n = len(labels)

labels += ["{:.0f}-CN-group(s) PAHs".format(k) for k in cn_dist.keys()]

#fig,ax = plt.subplots()

ax = fig.add_subplot(gs[34:,52:])
ax.pie(sizes,labels = labels,autopct="%1.1f%%")

ax.set_title(
    "C. Distribution of the substituted PAH dataset", 
    bbox = dict(edgecolor = "black", facecolor = "none" ), **annotate)

#fig.savefig(path + "\\plot\\data_pie_subst.jpeg",bbox_inches = "tight",dpi=600)
#plt.show()
fig.savefig(path + "\\plot\\data_pie_all.jpeg",bbox_inches = "tight",dpi=600)

"""
### Pie chart for the distribution of CN
sizes = [v*100/cn for v in cn_dist.values()]
labels = ["{:.0f}-CN-group(s) PAHs".format(k) for k in cn_dist.keys()]

fig,ax = plt.subplots()

ax.pie(sizes,labels = labels,autopct="%1.1f%%")

fig.savefig("data_pie_cn.jpeg",bbox_inches = "tight",dpi=600)
"""
