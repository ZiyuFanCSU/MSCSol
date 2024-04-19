from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import rdBase, Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Lipinski
import math,re,os
import numpy as np
import pandas as pd
import os
import pandas as pd
from rdkit.Chem import MACCSkeys

smile = 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)C(O)C1'
all_properties = {}
mol = Chem.MolFromSmiles(smile)

## molecular weight
all_properties['mol_weight'] = Descriptors.MolWt(mol)

## LogP
logp_m = Descriptors.MolLogP(mol)
all_properties['logp_m'] = logp_m

## Topological polar surface area
tpsa_m = Descriptors.TPSA(mol)
all_properties['tpsa_m'] = tpsa_m

## hba
all_properties['hba'] = Lipinski.NumHAcceptors(mol)

## hbd
all_properties['hbd'] = Lipinski.NumHDonors(mol)

## rob
all_properties['rob'] = Lipinski.NumRotatableBonds(mol)

## NumAliphaticRings
all_properties['aliRings'] = Lipinski.NumAliphaticRings(mol)

## NumAromaticRing
all_properties['aroRings'] = Lipinski.NumAromaticRings(mol)

## FractionCSP3
all_properties['sp3'] = Lipinski.FractionCSP3(mol)

## returns the Labute ASA value for a molecule
all_properties['LASA'] = rdMolDescriptors.CalcLabuteASA(mol)

## chiral_center
all_properties['chiral_center'] = len(Chem.FindMolChiralCenters(mol))

## QED
all_properties['qed'] = QED.qed(mol)

## MACCS
all_properties['MACCS'] = list(MACCSkeys.GenMACCSKeys(mol))

## ECFP6_1
arr = np.empty((0,2048), int).astype(int)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 1)
array = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, array)
all_properties['ECFP6_1'] = np.vstack((arr, array)).tolist()[0]

## ECFP6_2
arr = np.empty((0,2048), int).astype(int)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2)
array = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, array)
all_properties['ECFP6_2'] = np.vstack((arr, array)).tolist()[0]

## ECFP6_3
arr = np.empty((0,2048), int).astype(int)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3)
array = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, array)
all_properties['ECFP6_3'] = np.vstack((arr, array)).tolist()[0]

print(all_properties)





