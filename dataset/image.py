import json
from rdkit import Chem
from rdkit.Chem import Draw
import csv

mol = Chem.MolFromSmiles("CCCCNNN")
Draw.MolToFile(mol, f'./dataset/demo.png', size=(224, 224))
