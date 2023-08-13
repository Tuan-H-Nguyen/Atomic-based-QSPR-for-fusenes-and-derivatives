#%%
import os
import numpy as np
import pandas as pd
import glob

import os
import sys

DEFAULT_PATH = os.getcwd().split("\\")
try:
    DEFAULT_PATH = "\\".join(DEFAULT_PATH[:DEFAULT_PATH.index("data") + 1])
except: DEFAULT_PATH = ""

def get_total_data(data_type, prefix_path = "", dropna = True):
    """
    Get all data of PAH, thienoacenes, cyano-PAH and nitro-PAH
    """
    data_list = [        
            pd.read_csv(prefix_path + "raw_cyano_data.csv"),
            pd.read_csv(prefix_path + "raw_nitro_data.csv"),
            pd.read_csv(prefix_path + "raw_pah_data.csv")
        ]
    if data_type == "subst":
        total_data = pd.concat(data_list[:2])
    elif data_type == "pah":
        total_data = pd.concat(data_list[-1:])
    elif data_type == "mixed":
        total_data = pd.concat(data_list)

    if dropna:
        total_data = total_data.dropna()

    total_data = total_data.reset_index(drop=True)
    return total_data

def CN_count(smiles):
    if not "N" in smiles or "O" in smiles:
        return 0

    return smiles.count("N")

def NO2_count(smiles):
    if not "N" in smiles or "O" not in smiles:
        return 0

    return smiles.count("N")

def S_count(smiles):
    if "s" not in smiles and "S" not in smiles:
        return 0

    return smiles.count("S") + smiles.count("s")

intervals = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
class ReducedData:
    """
    Provide balance datasets
    Args:
        + subst_only: provide True to get substituted PAH dataset,
            provide False to get mixed or PAH dataset
        + pah_ only:  provide True to get PAH dataset
            provide False to get mixed or substituted PAH dataset
        note: if both arguments are False, a mixed dataset will be returned
    """
    def __init__(
        self,N,seed, path = DEFAULT_PATH,
        subst_only = False,
        pah_only = False
        ):

        np.random.seed(seed)
        self.seed_list = np.random.randint(0,1e6,N)
        self.n = 0

        cyano_path = path +"/raw_cyano_data.csv"
        try: self.cyano_data = pd.read_csv(cyano_path).dropna()
        except FileNotFoundError:
            print(cyano_path)
            path = input("The provided path is not found. New path to data?")
            cyano_path = path +"/raw_cyano_data.csv"
            self.cyano_data = pd.read_csv(cyano_path).dropna()

        nitro_path = path +"/raw_nitro_data.csv"
        self.nitro_data = pd.read_csv(nitro_path).dropna()

        pah_path = path +"/raw_pah_data.csv"
        self.pah_data = pd.read_csv(pah_path).dropna()

        assert not (subst_only and pah_only)
        self.subst_only = subst_only
        self.pah_only = pah_only

    def __call__(self):
        seed = self.seed_list[self.n]
        self.n += 1

        def concat_subs_data(
            data,count_fn, min_samples = 25,
            range_subs = range(1,5)):

            data_concat = []
            data.loc[:,"subs"] = data.loc[:,"smiles"].apply(count_fn)
            for i,sub in enumerate(range_subs):
                if i == len(range_subs) - 1:
                    subs_data = data.loc[data.subs >= sub]
                else:
                    subs_data = data.loc[data.subs == sub]

                assert (len(subs_data) > 0)
                data_concat.append(stratified_sampling(
                    min(min_samples,len(subs_data)), subs_data,
                    intervals = intervals, random_state = seed))
            data = pd.concat(data_concat,axis = 0)
            return data

        data_list = []
        if not self.pah_only:
            data = self.cyano_data
            cn_data = concat_subs_data(
                data,CN_count,
                min_samples = 150 if self.subst_only else 30
                )
            data_list.append(cn_data)

            data = self.nitro_data
            no2_data = concat_subs_data(
                data,NO2_count,
                min_samples = 150 if self.subst_only else 30
                )
            data_list.append(no2_data)

        if not self.subst_only:
            data = self.pah_data
            pah_data = concat_subs_data(
                data,S_count,
                min_samples = 200 if self.pah_only else 100,
                range_subs = [0,1])
            data_list.append(pah_data)

        data = pd.concat(data_list,axis = 0)
        data = data.drop("subs",axis = 1)
        data = data.reset_index(drop=True)
        return data

#get_total_data().to_csv("sample_DATA\\total.csv")

def stratified_sampling(
    sample_size,data,random_state,
    intervals = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    ):
    """
    Stratified sampling a data. Return a sub-dataset of the given 
    with specified size. The subdata are generated by 1/ binning the
    data according to the given intervals, 2/ sampling from each bin 
    round{[sample_size/(#samples all data)]*(#samples in bin)} samples,
    3/ concatenate data sampled from all bins
    Args:
    + sample_size (int): the size of the returned subset
    + data (pd.DataFrame): the data set from which subset is drawn.
    + intervals (list of float): [...,v_(i-1), v_i, v_(i+1)...] where 
        [v_(i-1),v_i] is a bin, [v_i,v_(i+1)] is a bin, ...
    + random_state (int): seed
    Return:
    + pd.DataFrame
    """
    np.random.seed(random_state)
    intervalSeeds = np.random.randint(np.iinfo(np.int32).max,size = len(intervals))
    #determine the number of samples of each interval
    intervalSize = [
        len(data.loc[(data.BG>intervals[i]) & (data.BG<intervals[i+1])]) 
        for i in range(len(intervals)-1)]
    #for each interval, pick a number of sample equal number of 
    #samples in that interval scale with quotient of sample size and 
    # data size and put all into one data set
    sampling = pd.concat([data.loc[
            (data.BG>intervals[i]) & (data.BG<intervals[i+1])
            ].sample(
                n=round(intervalSize[i]*(sample_size/len(data))),
                random_state = intervalSeeds[i])
            for i in range(len(intervals)-1)])
    #determine gap of sampled set and desired data set
    add_on = sample_size - len(sampling)
    #if the set is lack, then sample randomly
    if add_on > 0:
        spare = data.drop(sampling.index,axis=0).sample(n=add_on)
        sampling = pd.concat([sampling,spare])
    elif add_on < 0:
        spare = sampling.sample(n = abs(add_on))
        sampling = sampling.drop(spare.index)
    return sampling
