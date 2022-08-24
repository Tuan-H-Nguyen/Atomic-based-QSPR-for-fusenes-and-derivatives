import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.EState import EStateIndices
from rdkit.Chem.EState import AtomTypes

class MorganGen:
    def __init__(self,radius,nbits):
        self.radius = radius
        self.nbits = nbits
        print("ECFP fingerprint, radius = {}, nbits = {}".format(radius,nbits))

    def vectorize(self,smi):
        mol = Chem.MolFromSmiles(smi)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,self.radius, nBits = self.nbits)

        return np.array(fp)

class MACCSGen:
    def __init__(self):
        print("MACCS key")

    def vectorize(self,smi):
        mol = Chem.MolFromSmiles(smi)
        macss = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(
            mol)
        macss = np.array(macss)
        return macss

def FingerprintMol(mol):
    """ 
    source: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/EState/Fingerprinter.py
    generates the EState fingerprints for the molecule

    Concept from the paper: Hall and Kier JCICS _35_ 1039-1045 (1995)
    two numeric arrays are returned:
    The first (of ints) contains the number of times each possible atom type is hit
    The second (of floats) contains the sum of the EState indices for atoms of
      each type.
    """
    if AtomTypes.esPatterns is None:
        AtomTypes.BuildPatts()
    esIndices = EStateIndices(mol)

    nPatts = len(AtomTypes.esPatterns)
    counts = np.zeros(nPatts, dtype=np.int64)
    sums = np.zeros(nPatts, dtype=np.float64)
    for i, (_, pattern) in enumerate(AtomTypes.esPatterns):
        matches = mol.GetSubstructMatches(pattern, uniquify=1)
        counts[i] = len(matches)
        for match in matches:
            sums[i] += esIndices[match[0]]
    return counts, sums

class EStateGen:
    def __init__(self):
        print("EState fingerprint")


    def vectorize(self,smi):
        mol = Chem.MolFromSmiles(smi)
        fp = FingerprintMol(mol)
        return np.array(fp).reshape(-1)




