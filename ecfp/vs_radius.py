import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-1]))
sys.path.append("\\".join(path[:-1]))

import time
import numpy as np
import pickle

from data.data import ReducedData, stratified_sampling
from utils.criterion import RMSD

from pipeline import model_getter, graph_getter, data_splitter, ECFPVectorizer

N = 10
data = "mixed"
elec_prop_list = ["BG","EA","IP"]
num_iters = list(range(0,11,2))
random_state = 2022

method_list = [
    "rr","gpr"
    ]

result = np.zeros((
    N,len(method_list)*len(elec_prop_list),
    len(num_iters)))

if data == "mixed":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = False)
elif data == "pah":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = True, subst_only = False)
elif data == "subst":
    data_generator = ReducedData(
        N = 1000, seed = random_state, path = "data",
        pah_only = False, subst_only = True)

for i in range(N):
    print("Run #: {}".format(i))

    train_set, test_set = data_splitter(
        data_generator, train_split = 0.7, random_state = random_state)

    for j, method in enumerate(method_list):

        for k,num_iter in enumerate(num_iters):
            regress_model = method
            vectorizer = ECFPVectorizer(num_iter = num_iter)

            train_X = vectorizer.transform(train_set.loc[:,"smiles"])
            test_X = vectorizer.transform(test_set.loc[:,"smiles"])

            for e,elec_prop in enumerate(elec_prop_list):
                train_Y = np.array(train_set.loc[:,elec_prop])
                test_Y = np.array(test_set.loc[:,elec_prop])

                regressor = model_getter(regress_model,grid_search=True)

                regressor.fit(train_X,train_Y)

                test_Y_hat = regressor.predict(test_X)
                test_rmsd = RMSD(test_Y_hat, test_Y)

                #print(e+j*len(elec_prop_list))

                result[i,e+j*len(elec_prop_list), k] += test_rmsd 

with open("\\".join(path) + "\\vs_num_iter_"+data+".pkl","wb") as handle:
    pickle.dump(result,handle)



