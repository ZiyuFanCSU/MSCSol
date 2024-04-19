import csv,os
import pandas as pd
import numpy as np
def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])

        data = []
        for line in lines[2:]:
            atom_info = line.split()
            atom_symbol = atom_info[0]
            atom_coord = [float(coord) for coord in atom_info[1:4]]
            data.append([atom_symbol] + atom_coord)
        
        return num_atoms, np.array(data)

#parameters to change
input_file="/data.csv" #input file
output_file_dir="./Gaussian-data/Output_files_1" #where to put input files
commands="#PM6 opt freq geom=nocrowd" #Gaussian 09 commands to include
time="1:00:00" #time limit for calculation
data=pd.read_csv(input_file)#read input file
smi2=data["smiles"] #column containing SMILES
name_all=data["id"] #column containing StdInChIKey
for j in range(len(smi2)): #for each molecule
    name = str(name_all[j])
    smi = smi2[j]
    filename = "./xyz_1/"+name+".xyz"
    num_atoms, xyz_data = read_xyz(filename)
    #open a .com Gaussian 09 input file and add details
    f = open (output_file_dir + "/" + name + ".com","a")
    f.write("%nprocshared=4\n")
    f.write("%mem=100MW\n")
    f.write("%chk=" + name + ".chk\n")
    f.write(commands + "\n")
    f.write("\ngas opt\n\n")
    f.write("0 1\n")
    for atom in xyz_data:
        atom_symbol = atom[0]
        atom_coord = atom[1:]
        line = f"{atom_symbol:<2} {float(atom_coord[0]):>9.4f} {float(atom_coord[1]):>9.4f} {float(atom_coord[2]):>9.4f}\n"
        f.write(line)
    f.write('\n')
    f.write('\n')
    f.close()

    #open a .sh bash file for command line submission
    f = open (output_file_dir + "/" + name + ".sh","a")
    f.write("#$ -cwd -V\n")
    f.write("#$ -l h_vmem=4G\n")
    f.write("#$ -l h_rt=" + time + "\n")
    f.write("#$ -pe smp 4\n")
    f.write("#$ -m be\n")
    f.write("g09 " + name + ".com")
    f.close()