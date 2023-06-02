import argparse
import torch
import csv
import gvp.MSCSolmodel
model = gvp.MSCSolmodel.Model((7, 2), (100, 16), (32, 1), (32, 1))
print(model)
import features.data
from datetime import datetime
import tqdm, os, json
import numpy as np
import torch_geometric
import json
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default='./models/',
                    help='directory to save trained models, default=./models/')
parser.add_argument('--num-workers', metavar='N', type=int, default=0,
                   help='number of threads for loading data, default=4')
parser.add_argument('--max-nodes', metavar='N', type=int, default=32,
                    help='max number of per batch, default=32')
parser.add_argument('--epochs', metavar='N', type=int, default=1000,
                    help='training epochs, default=1000')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')

args = parser.parse_args()

node_dim = (100, 16) #self.so和self.vo
edge_dim = (32, 1)

print = partial(print, flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)
model_id = int(datetime.timestamp(datetime.now()))
dataloader = lambda x: torch_geometric.loader.DataLoader(x, 
                        num_workers=args.num_workers,drop_last=True,
                        batch_size = args.max_nodes)

def main():
    
    dataset = []
    solubility = []

    kf = StratifiedShuffleSplit(n_splits=10, train_size=0.9, test_size=0.1, random_state=args.seed)
    
    for train_index, valid_index in kf.split(list(range(0, len(dataset))), solubility):
        trainset, valset = dataset[train_index], dataset[valid_index]
        model = gvp.MSCSolmodel.Model((7, 2), node_dim, (32, 1), edge_dim).to(device)
        
        trainset = features.data.GraphDataset(trainset,img_transformer='train')
        valset = features.data.GraphDataset(valset,img_transformer='val')

        train(model, trainset, valset)
    
    
def train(model, trainset, valset):
    train_loader, val_loader = map(dataloader,(trainset, valset))
    optimizer = torch.optim.Adagrad(model.parameters(),lr=1e-3) # 

    for epoch in range(args.epochs):
        model.train()
        loss = loop(model, train_loader, optimizer=optimizer)
        print(f'EPOCH {epoch} TRAIN loss: {loss}')
        model.eval()
        with torch.no_grad():
            loss = loop(model, val_loader)    
        print(f'EPOCH {epoch} VAL loss: {loss}')

def loop(model, dataloader, optimizer=None):

    t = tqdm.tqdm(dataloader)   
    loss_fn = torch.nn.SmoothL1Loss()
    total_loss, total_count = 0, 0
    
    for batch in t:
        if optimizer: 
            optimizer.zero_grad()
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        logits = model(h_V, batch.edge_index, h_E, seq=batch.seq, batch33 = batch.batch, feature_molecule=batch.feature_molecule, picdata = batch.picdata)
        solu = batch.solu
        logits = torch.squeeze(logits)
        loss_value = loss_fn(logits, solu)
        if optimizer:
            loss_value.backward()
            optimizer.step()
        total_loss += float(loss_value)
        total_count += 1
        torch.cuda.empty_cache()

    return total_loss / total_count
    
if __name__== "__main__":
    main()
