import numpy as np
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device) 
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count 
    D_expand = torch.unsqueeze(D, -1) 
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class GraphDataset(data.Dataset):

    def __init__(self, data_list,
                 num_positional_embeddings=16, ratio = 1.0,
                 top_k=2, num_rbf=16, device="cpu",normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),img_transformer=None):
        super(GraphDataset, self).__init__()

        self.normalize = normalize
        self.ratio = ratio 
        self.type = img_transformer
        self._image_transformer_train = transforms.Compose([transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                 transforms.ToTensor()])
        self._image_transformer_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(list(e['atom_seq'].split(" "))) for e in data_list]
        self.letter_to_num=  ['C', 'O', 'H', 'N', 'S', 'P', 'Br', 'Cl', 'F', 'Si', 'Mn', 'Se', 'Li', 'Zn', 'Na', 'K', 'As', 'I', 'Fe', 'Ca',
                              'La', 'Cs', 'Cu', 'Cr', 'Ni', 'Mg', 'In', 'Co', 'Rh', 'Y', 'Pd', 'Mo', 'Au', 'Sm', 'V', 'Cd', 'B', 'Ba', 'Pb',
                               'Sr', 'Sn', 'Ru', 'Hg', 'Al', 'Ti', 'Ce', 'Ag', 'W', 'Re', 'Pt', 'Ge', 'Bi', 'Zr', 'Te', 'Gd', 'Lu', 'Pr',
                                'Sb', 'Nd', 'Dy', 'Nb', 'Ta', 'Hf', 'Be']
        self.letter_to_num = {v:i for i,v in enumerate(self.letter_to_num)}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

    def __len__(self): return len(self.data_list)

    def get_image(self, index):
        filename = index
        img = Image.open(filename).convert('RGB')
        if self.type == 'train':
            return self._image_transformer_train(img)
        else:
            return self._image_transformer_test(img)

    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, drug):
        name = drug['smiles']
        id = drug['id']
        classes = [0 if drug['LogS'] >= -1 else 1 if drug['LogS'] >= -3 else 2][0]
        
        img_list = ['img/','img_augment/']
        img_weights = [self.ratio, 1-self.ratio]
        selected_img = random.choices(img_list, weights=img_weights, k=1)[0]
        picdata = self.get_image("./dataset/"+selected_img+str(id)+".png")
        if self.normalize is not None:
            picdata = self.normalize(picdata)
        feature_molecule = drug['features']
        with torch.no_grad():
            coords_list = ['xyz','xyz_augment']
            coo_weights = [self.ratio, 1-self.ratio]
            selected_xyz = random.choices(coords_list, weights=coo_weights, k=1)[0]
            coords = torch.as_tensor(drug[selected_xyz],
                                     device=self.device, dtype=torch.float32)
            seq = torch.as_tensor([self.letter_to_num[a] for a in drug['atom_seq'].split(" ")],
                                  device=self.device, dtype=torch.long)
            solu = torch.as_tensor(drug['LogS'],
                                  device=self.device, dtype=torch.float32)
            X_ca = coords
            edge_index0 = torch_cluster.knn_graph(X_ca, k=self.top_k) 
            
            pos_embeddings = self._positional_embeddings(edge_index0)
            E_vectors = X_ca[edge_index0[0]] - X_ca[edge_index0[1]]  
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf,
                       device=self.device)  

            dihedrals = self._dihedrals(X_ca)  
            orientations = self._orientations(X_ca)

            b = self._onehot(seq)
            node_s = torch.cat([dihedrals, b], 1)
            node_v = orientations 
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1) 
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))

        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,solu = solu,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index0,feature_molecule=feature_molecule,id = id,picdata=picdata.unsqueeze(0),classes = classes)
        return data

    def _onehot(self,seq):
        all_one_hot = np.eye(5)
        seqseq = []
        seq = seq.tolist()
        seq2 = []
        for i in seq:
            if int(i) < 4:
                seq2.append(i)
            else:
                seq2.append(4)
        seqseq.append(seq2)
        b = torch.tensor(all_one_hot[seq2], dtype=torch.float32)
        return b

    def _dihedrals(self, X, eps=1e-7):
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2] 
        u_1 = U[1:-1] 
        u_0 = U[2:] 

        n_2 = _normalize(torch.cross(u_2, u_1),
                         dim=-1)  
        n_1 = _normalize(torch.cross(u_1, u_0),
                         dim=-1) 

        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 1])
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec