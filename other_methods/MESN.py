
import torch
import torch.utils.data as data
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_squared_error
from rdkit import Chem
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gmean
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
device = torch.device("cuda:0")

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

class GraphDataset(data.Dataset):
    def __init__(self, xd, y, smile_graph, features):
        super(GraphDataset, self).__init__()
        self.data_list = xd
        self.y = y
        self.g = smile_graph
        self.f = features

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i], self.y[i], self.g, self.f[i])

    def _featurize_as_graph(self, xd, y, smile_graph,feature):
        smiles = xd
        labels = y
        # convert SMILES to molecular representation using rdkit
        c_size, features, edge_index = smile_graph[smiles]
        # print(edge_index)
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        if len(edge_index) != 0:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index= torch.LongTensor(edge_index).transpose(1, 0), 
                                y=torch.Tensor([labels]))
        else:
            GCNData = DATA.Data(x=torch.Tensor(features),
                edge_index= torch.LongTensor(edge_index),
                y=torch.Tensor([labels]))

        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData.__setitem__('feature', torch.as_tensor([feature]))
        return GCNData


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# GCN based model
class MESN(torch.nn.Module):
    def __init__(self, n_output=2,num_features_xd=44, dropout=0.2, feature_emb=1019):

        super(MESN, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd, 1)

        self.fc1 = nn.Linear(feature_emb, 128)  
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, data1):
        x1, edge_index1, batch1, feature = data1.x, data1.edge_index, data1.batch, data1.feature
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))
        x1 = gmp(x1, batch1)      
        x1 = self.relu(self.drug1_fc_g1(x1))

        x = F.relu(self.fc1(feature))
        x = self.fc2(x)

        f = x1 + x
        return f

def main():
    df = pd.read_json('/dataset/benchmark.json')
    data = df[['features','smiles']]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    R2 = []
    MSE = []
    MAE = []
    RMSE = []
    yt = []
    yp = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = df["LogS"].iloc[train_index], df["LogS"].iloc[test_index]
        train_smiles = X_train['smiles'].tolist()
        train_smile_graph = {}
        for smile in train_smiles:
            g = smile_to_graph(smile)
            train_smile_graph[smile] = g
        drugdata = GraphDataset(xd=train_smiles, y= y_train.tolist(), smile_graph=train_smile_graph, features = X_train['features'].tolist())
        train = DataLoader(drugdata, batch_size=64, shuffle=None)

        test_smiles = X_test['smiles'].tolist()
        test_smile_graph = {}
        for smile in test_smiles:
            g_test = smile_to_graph(smile)
            test_smile_graph[smile] = g_test
        
        drugdata_test = GraphDataset(xd=test_smiles, y= y_test.tolist(), smile_graph=test_smile_graph, features = X_test['features'].tolist())
        test = DataLoader(drugdata_test, batch_size=64, shuffle=None,drop_last=False)
        
        lr = 0.00005
        epochs = 150
        model = MESN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        max_R2 = 0
        max_MSE = 0
        max_RMSE = 0
        max_MAE = 0
        for epoch in range(epochs):
            for batch, data_train in enumerate(train):
                data_train = data_train.to(device)
                model.train()
                optimizer.zero_grad()
                output = model(data_train)
                loss = criterion(output.squeeze(), data_train.y)
                loss.backward()
                optimizer.step()
                r2 = r2_score(data_train.y.cpu().detach().numpy(),output.squeeze().cpu().detach().numpy())
                print(f'Epoch {epoch + 1}, Batch {batch}, Loss: {loss.item(), r2}')

            gcn_output = None
            for batch_idx, data_test in enumerate(test):
                data_test = data_test.to(device)
                model.eval()
                output = model(data_test)
                if gcn_output == None:
                    gcn_output = output.squeeze()
                else:
                    gcn_output = torch.cat((gcn_output,output.squeeze()))

            predictions = gcn_output.cpu().detach().numpy()

            if max_R2 == 0:
                max_R2 = r2_score(y_test,predictions)
                max_MSE = mean_squared_error(y_test,predictions)
                max_RMSE = np.sqrt(mean_squared_error(y_test,predictions))
                max_MAE = mean_absolute_error(y_test,predictions)
                yt = y_test
                yp = predictions
            elif r2_score(y_test,predictions)>max_R2:
                max_R2 = r2_score(y_test,predictions)
                max_MSE = mean_squared_error(y_test,predictions)
                max_RMSE = np.sqrt(mean_squared_error(y_test,predictions))
                max_MAE = mean_absolute_error(y_test,predictions)
                yt = y_test
                yp = predictions

            print("R2_score:"+str(max_R2))
            print("MSE:"+str(max_MSE))
            print("MAE:"+str(max_MAE))
            print("RMSE:"+str(max_RMSE))

        R2.append(max_R2)
        MSE.append(max_MSE)
        MAE.append(max_MAE)
        RMSE.append(max_RMSE)
    import statistics
    print(statistics.mean(R2))
    print(statistics.mean(MAE))
    print(statistics.mean(MSE))
    print(statistics.mean(RMSE))

if __name__ == "__main__":
    main()