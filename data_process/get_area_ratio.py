"""
| Machine Learning with Physicochemical Relationships: Solubility Prediction in Organic Solvents and Water(Supplementary)
"""
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
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import os
import pandas as pd
from rdkit.Chem import MACCSkeys
#number of points per atom
number_points=1000
#vdw radii
radiiDict={'H': 1.2, 'He': 1.4, 'Li': 2.2, 'Be': 1.9, 'B': 2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.5,
  'Na': 2.4, 'Mg': 2.2, 'Al': 2.1, 'Si': 2.1, 'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Ar': 1.9, 'K': 2.8, 'Ca': 2.4,
    'Se': 1.9, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05, 'Mn': 2.05, 'Fe': 2.05, 'Co': 2.0, 'Ni': 2.0, 'Cu': 2.0, 'Zn': 2.1,
      'Ga': 2.1, 'Ge': 2.1, 'As': 2.05, 'Br': 1.85, 'Kr': 2.0, 'Rb': 2.9, 'Sr': 2.55, 'Y': 2.4, 'Zr': 2.3, 'Nb': 2.15,
        'Mo': 2.1, 'Tc': 2.05, 'Ru': 2.05, 'Rh': 2.0, 'Pd': 2.05, 'Ag': 2.1, 'Cd': 2.2, 'In': 2.2, 'Sn': 2.25, 'Sb': 2.2,
          'Te': 2.1, 'I': 1.98, 'Xe': 2.2, 'Cs': 3.0, 'Ba': 2.7, 'La': 2.5, 'Ce': 2.54, 'Pr': 2.56, 'Nd': 2.51, 'Pm': 2.54,
            'Sm': 2.5, 'Eu': 2.64, 'Gd': 2.54, 'Tb': 2.46, 'Dy': 2.49, 'Ho': 2.42, 'Er': 2.51, 'Tm': 2.47, 'Yb': 2.6,
              'Lu': 2.43, 'Hf': 2.3, 'Ta': 2.27, 'W': 2.2, 'Re': 2.2, 'Os': 2.17, 'Ir': 2.13, 'Pt': 2.07, 'Au': 2.07,
                'Hg': 2.07, 'Tl': 2.26, 'Pb': 2.21, 'Bi': 2.31, 'Po': 2.44, 'At': 2.44, 'Rn': 2.43, 'Fr': 3.1, 'Ra': 2.85,
                  'Ac': 2.84, 'Th': 2.56, 'Pa': 2.54, 'U': 2.41, 'Np': 2.36, 'Pu': 2.54, 'Am': 2.56, 'Cm': 2.56, 'Bk': 2.56,
                    'Cf': 2.56, 'Es': 2.56, 'Fm': 2.56}

## areas && asp_ratios
def get_points(xyz):
    #open .xyz file
    f = open(xyz,"r")
    data = []
    #get contents
    for line in f:
        data.append(line)
    #delete first 2 lines which do not contain Cartesians
    del(data[0])
    del(data[0])
    #make new lists to populate
    atom_pos=[]
    atom_names=[]
    atom_radii=[]
    #for each line (atomic XYZ coordinates)
    for f in range(len(data)):
        #remove line space
        data[f]=data[f].replace('\n','')
        #split into atom,x,y,z
        data[f]=re.split(r'\s+',data[f])
        #get atom name
        atom_names.append(data[f][0])
        #delete atom name
        del(data[f][0])
    #atom position is remaining x,y,z
    atom_pos=data
    #convert coordinates to floats
    atom_pos = [[float(j) for j in i] for i in atom_pos]
    #get the corresponding radius for each atom
    for f in range(len(atom_names)):
        atom_radii.append(radiiDict[atom_names[f]])
    #define new list
    data=[]
    #for each atom
    for f in range(len(atom_names)):
        #Produce random points in a cube
        x=((2*atom_radii[f])*np.random.rand(number_points,3))-atom_radii[f]
        #Keep points inside the sphere
        keep=[]
        for point in x:
            if math.sqrt(((point[0])**2)+((point[1])**2)+((point[2])**2)) < atom_radii[f]:
                keep.append(point)
        keep=np.array(keep)
        #Project points to surface of sphere
        x1=[]
        y1=[]
        z1=[]
        for point in keep:
            d=math.sqrt(((point[0])**2)+((point[1])**2)+((point[2])**2))
            scale=(atom_radii[f]-d)/d
            point=point+(scale*point)
            x1.append(point[0])
            y1.append(point[1])
            z1.append(point[2])
        #Move atom to correct position
        for i in range(len(x1)):
            x1[i]=x1[i]+atom_pos[f][0]
        for i in range(len(y1)):
            y1[i]=y1[i]+atom_pos[f][1]
        for i in range(len(z1)):
            z1[i]=z1[i]+atom_pos[f][2]
        data.append(x1)
        data.append(y1)
        data.append(z1)
    #Discard points in shape interior
    for f in range(len(atom_names)):
        for g in range(len(atom_names)):
            if g==f:
                continue
            keep=[]
            for i in range(len(data[3*f])):
                if math.sqrt(((data[3*f][i]-atom_pos[g][0])**2)+((data[(3*f)+1][i]-atom_pos[g][1])**2)+((data[(3*f)+2][i]-atom_pos[g][2])**2)) > atom_radii[g]:
                    keep.append(i)
            x1_keep=[]
            y1_keep=[]
            z1_keep=[]
            for x in keep:
                x1_keep.append(data[3*f][x])
                y1_keep.append(data[(3*f)+1][x])
                z1_keep.append(data[(3*f)+2][x])
            data[(3*f)]=x1_keep
            data[(3*f)+1]=y1_keep
            data[(3*f)+2]=z1_keep
    x=[]
    y=[]
    z=[]
    #merge points
    for f in range(len(data)):
        if f%3 == 0:
            for g in data[f]:
                x.append(g)
        if f%3 == 1:
            for g in data[f]:
                y.append(g)
        if f%3 == 2:
            for g in data[f]:
                z.append(g)
    #return separate x, y and z point lists
    return(x,y,z)

#plot and save graphs
def graph(x,y,z,az,el):
    fig = plt.figure(figsize=(20,20)) #large canvas size for resolution and to fit larger molecules
    #use 3d plotting
    ax = fig.add_subplot(111,projection='3d')
    #colour black with big point size so image opaque
    ax.scatter(x,y,z,color="black",s=100)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_zlim(-20,20)
    #no axes!
    plt.axis('off')
    #axim and alev are the angles to define the view, change to get projection down each axis
    ax.view_init(azim=az, elev=el)

def shadow_info(image):
    #initiate list for results
    results=[]
    #load image
    img = cv.imread(image,0)
    #fit shape
    ret,thresh = cv.threshold(img,127,255,0)
    #get shape by increasing area
    contours,hierarchy=cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #get second last (last is full frame)
    cnt=contours[-2]
    #get area
    area=cv.contourArea(cnt)
    results.append(area)
    #fit minimum area rectangle
    rect = cv.minAreaRect(cnt)
    #width,height,area and aspect ratio
    width = float(rect[1][0])
    length = float(rect[1][1])
    rect_area=width*length
    aspect_ratio = width/length
    #do not know whether width or height is larger
    if aspect_ratio>1:
        aspect_ratio = length/width
    results.append(aspect_ratio)
    return(results)
#save x, y, z surface points
dir1="./dataset/data_process/xyz_1"
dir2="./dataset/data_process/vdw_points"

for files in os.listdir(dir1):
    x,y,z = get_points(dir1 + "/" + files)
    files=files.replace(".xyz","")
    f = open(dir2 + "/" + files + ".csv","a")
    for i in range(len(x)):
        f.write(str(x[i]) + "," + str(y[i]) + "," + str(z[i]) + "\n")
    f.close()
    
#reload surface points file and save images
#new directory to place images
dir3="./dataset/data_process/shadow_image_vdw"

for files in os.listdir(dir2):
    data=np.loadtxt(dir2 + "/" + files,delimiter=",")
    files=files.replace(".csv","")
    x=data[:,0]
    y=data[:,1]
    z=data[:,2]
    graph(x,y,z,0,0)
    plt.savefig(dir3 + "/" + files + "_1.png") #first angle
    plt.close()
    graph(x,y,z,90,0)
    plt.savefig(dir3 + "/" + files + "_2.png") #perpendicular
    plt.close()
    graph(x,y,z,90,90)
    plt.savefig(dir3 + "/" + files + "_3.png") #perpendicular again
    plt.close()

#get for every molecule
dir4="./dataset/shadow_desc.csv"
data=[]
for files in os.listdir(dir1):
    row=[]
    files=files.replace(".xyz","")
    results1=shadow_info(dir3 + "/" + files + "_1.png")
    results2=shadow_info(dir3 + "/" + files + "_2.png")
    results3=shadow_info(dir3 + "/" + files + "_3.png")
    areas=sorted([results1[0],results2[0],results3[0]]) #get ascending areas
    asp_ratios=sorted([results1[1],results2[1],results3[1]]) #get ascending aspect ratios
    row.append(files) #get names
    for ar in areas: #add areas
        row.append(ar)
    for asp in asp_ratios: #add aspect ratios
        row.append(asp)
    data.append(row)
df=pd.DataFrame(data=data,columns=["id","Area1","Area2","Area3","Asp1","Asp2","Asp3"])
df.to_csv(dir4,index=False)
print(df)