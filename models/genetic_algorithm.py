import time
import argparse
import numpy as np
import pickle
from copy import deepcopy

from data import ReducedData, split_data, get_total_data
from model_kr_wl import WL_model
from wl_kernel import generate_fpgen, FingerprintGenerator, WLSubtreeEdge, WLEdge, WLShortestPath
import pygad

data_type = "mixed"
model_type = "rr"
kernel_type = "shortestPath"
radius = 2

num_gens = 50

def get_data(data_type):
    if data_type == "subst":
        data = ReducedData(500,500,subst_only=True)
    elif data_type == "pah":
        data = ReducedData(500,500,pah_only=True)
    elif data_type == "mixed":
        data = ReducedData(500,500)

    total = data()
    return total 

def gene_to_fpgen(fpgen,gene):
    assert len(gene) == len(fpgen.unique_hash)

    labels = np.array(fpgen.unique_hash)
    labels = labels[np.array(gene) == 1]

    fpgen = deepcopy(fpgen)
    fpgen.unique_hash = labels

    return fpgen

class FitnessFunc:
    def __init__(self,radius, model_type, data_type = "subst", kernel_type = "shortestPath"):
        total = get_total_data(data_type)
        self.master_fpgen = generate_fpgen(total,radius = radius, kernel_type = kernel_type)
        self.model_type = model_type

        if data_type == "pah":
            self.data = ReducedData(500,500,pah_only = True, subst_only = False)
        elif data_type == "subst":
            self.data = ReducedData(500,500,pah_only = False, subst_only = True)
        elif data_type == "mixed":
            self.data = ReducedData(500,500,pah_only = False, subst_only = False)
        
    def __call__(self,solution,solution_idx):
        fpgen = gene_to_fpgen(self.master_fpgen, solution)

        model, error = WL_model(
            self.data() ,test = None, fpgen = fpgen,
            cv = 2, model = self.model_type , verbose = False,
            grid_search_mode = "min"
            )

        return np.mean(error)

def load_ga_fpgen(
    model_type, radius, data_type, fitness_func,
    kernel_type = "shortestPath", num_gens = 50
    ):

    #file_name = "GA_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
    #file_name = "GAresult_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
    file_name = "GA" + str(num_gens) + "_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
    ga_instance = pygad.load(file_name)

    best_sol ,_, _ = ga_instance.best_solution()

    fpgen = gene_to_fpgen(fitness_func.master_fpgen, best_sol)

    return fpgen


def load_ga_fpgen_from_list(
    model_type, radius, data_type,
    kernel_type = "shortestPath", num_gens = 50
    ):
    #file_name = "listGA_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type + ".pkl"
    #file_name = "GAresult_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
    file_name = "listGA" + str(num_gens) + "_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type

    with open(file_name,"rb") as f:
        unique_hashes = pickle.load(f)

    fpgen = FingerprintGenerator()
    fpgen.unique_hash = list(unique_hashes)

    if kernel_type == "subtree": fpgen.wl_class = WLSubtree
    elif kernel_type == "edge": fpgen.wl_class = WLEdge
    elif kernel_type == "subtreeEdge": fpgen.wl_class = WLSubtreeEdge
    elif kernel_type == "shortestPath": 
        fpgen.wl_class = WLShortestPath

    fpgen.radius = radius

    return fpgen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c","--convert", type = int, choices = [0,1],
        default = 0)

    args = parser.parse_args()

    print(data_type)
    print(model_type)
    print(kernel_type)
    print(radius)
    print("num_gens: ",num_gens)

    if args.convert:
        file_name = "GA" + str(num_gens) + "_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
        fitness_func = FitnessFunc(data_type = data_type, model_type = model_type, radius = radius)

        fpgen = load_ga_fpgen(model_type,radius,data_type,fitness_func,kernel_type)

        #file_name = "listGA_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type + ".pkl"
        file_name = "listGA" + str(num_gens) + "_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type + ".pkl"

        with open(file_name,"wb") as f:
            pickle.dump(list(fpgen.unique_hash),f)

    else:
        num_generations = num_gens
        num_parents_mating = 4

        fitness_function = FitnessFunc(data_type = data_type, model_type = model_type, radius = radius)
        def fitness_func(sol,sol_idx):
            start = time.time()
            fitness = fitness_function(sol,sol_idx)
            print("fitness calculation runtime:",time.time() - start)
            return fitness

        sol_per_pop = 8
        num_genes = len(fitness_function.master_fpgen.unique_hash)

        init_range_low = 0
        init_range_high = 2
        gene_type = int

        parent_selection_type = "sss"
        keep_parents = 1

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 10

        ga_instance = pygad.GA(
            num_generations=num_generations,
           num_parents_mating=num_parents_mating,
           fitness_func=fitness_func,
           sol_per_pop=sol_per_pop,
           num_genes=num_genes,
           init_range_low=init_range_low,
           init_range_high=init_range_high,
           gene_type = gene_type,
           parent_selection_type=parent_selection_type,
           keep_parents=keep_parents,
           crossover_type=crossover_type,
           mutation_type=mutation_type,
           mutation_percent_genes=mutation_percent_genes,
           )

        start = time.time()
        ga_instance.run()
        print("Runtime:", time.time() - start)

        file_name = "GA" + str(num_gens) + "_" + model_type + "_" +kernel_type+str(radius)+"_"+data_type
        ga_instance.save(file_name)
