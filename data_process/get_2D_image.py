import json
from rdkit import Chem
from rdkit.Chem import Draw
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd


df = pd.read_csv("./data.csv") 

for index, row in df.iterrows():
    smiles = row['smiles']
    id = row['id']
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol,f'./img/'+str(id)+'.png',size=(224, 224))

for index, row in df.iterrows():
    smiles = row['smiles']
    id = row['id']
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    Draw.MolToFile(mol,f'./img_H/'+str(id)+'.png',size=(224, 224))

import os
import random
from PIL import Image

min_angle = -180
max_angle = 180

folder_path = "./img" 

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        angle = random.randint(min_angle, max_angle)

        rotated_image = image.rotate(angle, expand=False)
        rotated_image.save(os.path.join("./img_rotate", f"rotated_{filename}"))


for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
        flipped_image = image.transpose(flip)
        flipped_image.save(os.path.join("./img_trans", f"flipped_{filename}"))
