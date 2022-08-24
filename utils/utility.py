import numpy as np
import torch

ELEC_PROP_FULL = {
    "BG": "Band Gap","EA":"Electron Affinity", "IP":"Ionization Potential"}

def zip_dict(dicts):
    result = []

    keys = list(dicts.keys())
    values = list(dicts.values())
    values = list(zip(*values))

    for v in values:
        d = {}
        for i,k in enumerate(keys):
            d.update({k:v[i]})
        result.append(d)
    return result
    

