import csv
import pandas as pd
import datetime
import numpy as np
import os
from rdkit.Chem import AllChem
from rdkit import Chem
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d

def getget_atom(smile):
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    atoms = []
    for atom in mol.GetAtoms(): 
        atoms.append(atom.GetSymbol())   
    return atoms

def smiletoGaussian09(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    data = mol_to_geognn_graph_data_MMFF3d(mol)
    data['smiles'] = smiles
    atomlist = getget_atom(smiles)

    filename = './demo/Gaussian-data/'+str(datetime.datetime.now().strftime('%d%H%m%s'))

    with open(filename + '.gjf', 'w') as file_object:
        file_object.write("%mem=1GB\n")
        file_object.write("%nprocshared=8\n")
        file_object.write("# apfd/6-311+G(d,p) geom=nocrowd\n")
        file_object.write("\n")
        file_object.write(filename + '\n')
        file_object.write("\n")
        file_object.write("0 1\n")
        sum = 0
        for i in atomlist:
            file_object.write(' ' + i)
            if data['atom_pos'][sum][0] < 0:
                for num in range(18 - len(i)):
                    file_object.write(" ")
                if len(str(data['atom_pos'][sum][0])) <= 11:
                    file_object.write(str(data['atom_pos'][sum][0]))
                    for num2 in range(11 - len(str(data['atom_pos'][sum][0]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][0])[0:11])
            else:
                for num in range(19 - len(i)):
                    file_object.write(" ")
                if len(str(data['atom_pos'][sum][0])) <= 10:
                    file_object.write(str(data['atom_pos'][sum][0]))
                    for num2 in range(10 - len(str(data['atom_pos'][sum][0]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][0])[0:10])

            if data['atom_pos'][sum][1] < 0:
                file_object.write("   ")
                if len(str(data['atom_pos'][sum][1])) <= 11:
                    file_object.write(str(data['atom_pos'][sum][1]))
                    for num2 in range(11 - len(str(data['atom_pos'][sum][1]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][1])[0:11])
            else:
                file_object.write("    ")
                if len(str(data['atom_pos'][sum][1])) <= 10:
                    file_object.write(str(data['atom_pos'][sum][1]))
                    for num2 in range(10 - len(str(data['atom_pos'][sum][1]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][1])[0:10])

            if data['atom_pos'][sum][2] < 0:
                file_object.write("   ")
                if len(str(data['atom_pos'][sum][2])) <= 11:
                    file_object.write(str(data['atom_pos'][sum][2]))
                    for num2 in range(11 - len(str(data['atom_pos'][sum][2]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][2])[0:11])
            else:
                file_object.write("    ")
                if len(str(data['atom_pos'][sum][2])) <= 10:
                    file_object.write(str(data['atom_pos'][sum][2]))
                    for num2 in range(10 - len(str(data['atom_pos'][sum][2]))):
                        file_object.write("0")
                else:
                    file_object.write(str(data['atom_pos'][sum][2])[0:10])
            sum = sum + 1
            file_object.write("\n")
        file_object.write("\n")

    file_object.close()

    os.system('g09 %s.gjf' % filename)

    line_number = 0
    count = 0
    result = ' '
    for line in open(filename + ".log", "r", encoding='UTF-8'):
        if 'Dipole moment (field-independent basis, Debye):' in line:
            print(line_number)
            print(line)
            count = line_number
        if count == line_number - 1:
            print(line)
            result = line
        line_number = line_number + 1

    result = result.split()
    print(result)
    dipole = []
    dipole.append(smiles)
    for i in range(len(result)):
        if i % 2 != 0:
            dipole.append(result[i])

    return dipole


if __name__ == '__main__':
    smiles = "OCc1ccccc1CNCCCCCN"
    dipole = smiletoGaussian09(smiles)
    print(dipole)







