import torch
import torch.nn as nn
from gvp import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean
from .SKANet import SKAnet

class Model(torch.nn.Module):

    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
    
        super(Model, self).__init__()
        self.resnet18 = SKAnet.skresnext50_32x4d()
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 256)
        self._resnetrelu1 = nn.LeakyReLU(0.1)
        self._resnetpred4 = nn.Linear(256,128)
        self._resnetrelu2 = nn.LeakyReLU(0.1)
        self._resnetpred5 = nn.Linear(128,1)
        
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))

        node_h_dim0, node_h_dim1 = node_h_dim  
        self.shiliang0 = nn.Linear(48,24)
        self.shiliang1 = nn.Linear(24,1)

        self.pred0 = nn.Linear(int(node_h_dim0),50)
        self.pred1 = nn.Linear(50,1)
        self.pred2 = nn.Linear(1019,512) 
        self.relu0 = nn.LeakyReLU(0.1)
        self.pred3 = nn.Linear(512,256)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pred4 = nn.Linear(256,128)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pred5 = nn.Linear(128,1)
        self.pred = nn.Linear(4,1)

    def forward(self, h_V, edge_index, h_E, seq,batch33,feature_molecule,picdata):
        device = edge_index.device
        resnet18 = self.resnet18(picdata.clone().detach().to(device=device, dtype=torch.float32))
        logits21_resnet = self._resnetrelu1(resnet18)
        logits4_resnet = self._resnetpred4(logits21_resnet)
        logits22_resnet = self._resnetrelu2(logits4_resnet)
        logits5_resnet = self._resnetpred5(logits22_resnet)
        
        h_V = self.W_v(h_V) 
        h_E = self.W_e(h_E) 
        
        for layer in self.encoder_layers:  
            h_V = layer(h_V, edge_index, h_E)
        
        encoder_embeddings = h_V
        shapee = int(str(encoder_embeddings[1].shape).split("[")[1].split(',')[0])

        t_view = encoder_embeddings[1].reshape(shapee,48)
        shiliang = self.shiliang0(t_view)
        shiliang2 = self.shiliang1(shiliang)

        logits0 = self.pred0(encoder_embeddings[0])
        logits1 = self.pred1(logits0)
        out = scatter_mean(logits1, batch33, dim=0)

        logits2 = self.pred2(torch.tensor(feature_molecule,device=device, dtype=torch.float32))
        logits20 = self.relu0(logits2)
        logits3 = self.pred3(logits20)
        logits21 = self.relu1(logits3)
        logits4 = self.pred4(logits21)
        logits22 = self.relu2(logits4)
        logits5 = self.pred5(logits22)
        
        out2 = torch.cat([out, logits5], 1)
        out3 = torch.cat([out2, logits5_resnet], 1)
        shiliang3 = scatter_mean(shiliang2, batch33, dim=0)
        out4 = torch.cat([out3, shiliang3], 1)
        logits6 = self.pred(out4)

        return logits6

class Model_cl(torch.nn.Module):

    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1):
    
        super(Model_cl, self).__init__()
        self.resnet18 = SKAnet.skresnext50_32x4d()
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, 256)
        self._resnetrelu1 = nn.LeakyReLU(0.1)
        self._resnetpred4 = nn.Linear(256,128)
        self._resnetrelu2 = nn.LeakyReLU(0.1)
        self._resnetpred5 = nn.Linear(128,1)
        
        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))

        node_h_dim0, node_h_dim1 = node_h_dim
        self.shiliang0 = nn.Linear(48,24)
        self.shiliang1 = nn.Linear(24,1)

        self.pred0 = nn.Linear(int(node_h_dim0),50)
        self.pred1 = nn.Linear(50,1)
        self.pred2 = nn.Linear(1019,512)
        self.relu0 = nn.LeakyReLU(0.1)
        self.pred3 = nn.Linear(512,256)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pred4 = nn.Linear(256,128)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pred5 = nn.Linear(128,1)
        self.pred = nn.Linear(4,3)

    def forward(self, h_V, edge_index, h_E, seq,batch33,feature_molecule,picdata):
        device = edge_index.device
        resnet18 = self.resnet18(picdata.clone().detach().to(device=device, dtype=torch.float32))
        logits21_resnet = self._resnetrelu1(resnet18)
        logits4_resnet = self._resnetpred4(logits21_resnet)
        logits22_resnet = self._resnetrelu2(logits4_resnet)
        logits5_resnet = self._resnetpred5(logits22_resnet)
        
        h_V = self.W_v(h_V) 
        h_E = self.W_e(h_E) 
        
        for layer in self.encoder_layers:  
            h_V = layer(h_V, edge_index, h_E)
        
        encoder_embeddings = h_V
        shapee = int(str(encoder_embeddings[1].shape).split("[")[1].split(',')[0])

        t_view = encoder_embeddings[1].reshape(shapee,48)
        shiliang = self.shiliang0(t_view)
        shiliang2 = self.shiliang1(shiliang)

        logits0 = self.pred0(encoder_embeddings[0])
        logits1 = self.pred1(logits0)
        out = scatter_mean(logits1, batch33, dim=0)

        logits2 = self.pred2(torch.tensor(feature_molecule,device=device, dtype=torch.float32))
        logits20 = self.relu0(logits2)
        logits3 = self.pred3(logits20)
        logits21 = self.relu1(logits3)
        logits4 = self.pred4(logits21)
        logits22 = self.relu2(logits4)
        logits5 = self.pred5(logits22)
        
        out2 = torch.cat([out, logits5], 1)
        out3 = torch.cat([out2, logits5_resnet], 1)
        shiliang3 = scatter_mean(shiliang2, batch33, dim=0)
        out4 = torch.cat([out3, shiliang3], 1)
        logits6 = self.pred(out4)

        return logits6
    
