import argparse
import torch
import gvp.MSCSolmodel
import features.data
from datetime import datetime
import os, json
from tqdm import tqdm
import numpy as np
import torch_geometric
import json
from math import sqrt
import random
from functools import partial
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import logging
import time
from utils.utils import AverageMeter
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold
def create_logger(logger_file_name):
    """
    :param logger_file_name:
    :return:
    """
    logger = logging.getLogger()         
    logger.setLevel(logging.INFO)       

    file_handler = logging.FileHandler(logger_file_name)  
    console_handler = logging.StreamHandler()              
	
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )
    
    file_handler.setFormatter(formatter)     
    console_handler.setFormatter(formatter)   
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default='./models/',
                    help='directory to save trained models, default=./models/')
parser.add_argument('--show_progressbar', action='store_true')
parser.add_argument('--num_workers', metavar='N', type=int, default=1,
                   help='number of threads for loading data, default=1')
parser.add_argument('--scheduler_ornot', metavar='N', type=int, default=0,
                   help='about the scheduler')
parser.add_argument('--num_graph', metavar='N', type=int, default=32,
                    help='number of per batch, default=128')
parser.add_argument('--lr', metavar='LR', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num', metavar='N', type=int, default=128,
                    help='scheduler rule, default=100')
parser.add_argument('--ratio_for_coords', metavar='N', type=float, default=0.5,
                    help='ratio for augment')
parser.add_argument('--epochs', metavar='N', type=int, default=5000,
                    help='training epochs, default=1000')
parser.add_argument('--seed', type=int, default=42, 
                    help='random seed (default: 42) to split dataset')
parser.add_argument('--log_path', default='./results/logger.log', type=str, 
                    help='log file path to save result')
parser.add_argument('--c_or_l', default='l', type=str, 
                    help='Classification or Regression')
parser.add_argument('--model_name', default='default', type=str, 
                    help='Classification or Regression')
parser.add_argument('--device', default='cuda:0', type=str, choices=['cpu', 'cuda:0'],
                    help='Choose "cpu" to use the central processing unit or "gpu" for the graphics processing unit.')

args = parser.parse_args()
disable_tqdm = not args.show_progressbar
ratio = args.ratio_for_coords
logger = create_logger(args.log_path)

random.seed(args.seed)
node_dim = (100, 16)
edge_dim = (32, 1)

print = partial(print, flush=True)
device = torch.device(args.device)

if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)
model_id = int(datetime.timestamp(datetime.now()))
dataloader = lambda x: torch_geometric.loader.DataLoader(x, 
                        num_workers=args.num_workers,drop_last=True,
                        batch_size = args.num_graph)

def main():
    with open('./dataset/benchmark.json', 'r') as f:
        data = json.load(f)      

    dataset = data

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in range(5):
        LOSS = AverageMeter()
        MAE = AverageMeter()
        MSE = AverageMeter()
        RMSE = AverageMeter()
        R2 = AverageMeter()

        train_indices = []
        valid_indices = []

        for train_index, valid_index in kf.split(data):
            train_indices.append(train_index)
            valid_indices.append(valid_index)

        train_index, valid_index = train_indices[k], valid_indices[k]
        trainset = []
        valset = []
        for t in train_index:
            trainset.append(dataset[t])
        for v in valid_index:
            valset.append(dataset[v])
        if args.model_name == "default":
            model = gvp.MSCSolmodel.Model((7, 2), node_dim, (32, 1), edge_dim).to(device)
        else:
            model = gvp.MSCSolmodel.Model_cl(args.model_name).to(device)

        trainset = features.data.GraphDataset(trainset,img_transformer='train',ratio=ratio)
        valset = features.data.GraphDataset(valset,img_transformer='val',ratio=ratio)
        logger.info(f'----------Begin Training Model---------'
                    f'Fold: {k} ')
        test_Loss, test_MAE, test_MSE, test_RMSE, test_R2 = train(model, trainset, valset)
        LOSS.update(test_Loss)
        MAE.update(test_MAE)
        MSE.update(test_MSE)
        RMSE.update(test_RMSE)
        R2.update(test_R2)
        logger.info(f'5-fold cross-validation: '
                            f'Loss {LOSS.avg:.3f} '
                            f'MAE {MAE.avg:.3f}'
                            f'MSE {MSE.avg:.3f}'
                            f'RMSE {RMSE.avg:.3f}'
                            f'R2 {R2.avg:.3f}')
def make_rule(num):
    def rule(epoch):
        if epoch < num:
            return 1.0  
        else:
            steps_since_2000 = (epoch - num) // 5
            scale_factor = max(0.1, 1.0 - 0.09 * min(10, steps_since_2000))
            return scale_factor
    return rule
    
def train(model, trainset, valset):
    train_loader = dataloader(trainset)
    val_loader = dataloader(valset)
    optimizer = torch.optim.Adagrad(model.parameters(),lr=args.lr)
    if args.scheduler_ornot == True:
        lr_rule = make_rule(args.num)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_rule)
    else:
        scheduler = None

    test_MAE_max, test_MSE_max, test_RMSE_max, test_R2_max = None, None, None, None
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_Loss, train_MAE, train_MSE, train_RMSE, train_R2 = loop(epoch, model, train_loader,mode="train", optimizer=optimizer, scheduler=scheduler)
        logger.info(f'Train: '
                        f'Loss {train_Loss:.3f} '
                        f' MAE {train_MAE:.3f}'
                        f' MSE {train_MSE:.3f}'
                        f' RMSE {train_RMSE:.3f}'
                        f' R2 {train_R2:.3f}')
        model.eval()
        end_time = time.time()
        time_time = end_time - start_time
        print(time_time)
        with torch.no_grad():
            test_Loss, test_MAE, test_MSE, test_RMSE, test_R2 = loop(epoch, model, val_loader)
            if test_R2_max == None:
                test_MAE_max, test_MSE_max, test_RMSE_max, test_R2_max = test_MAE, test_MSE, test_RMSE, test_R2
            else:
                if test_R2 > test_R2_max:
                    test_MAE_max, test_MSE_max, test_RMSE_max, test_R2_max = test_MAE, test_MSE, test_RMSE, test_R2
            logger.info(f'Valid: '
                        f'Loss {test_Loss:.3f} '
                        f' MAE {test_MAE:.3f}'
                        f' MSE {test_MSE:.3f}'
                        f' RMSE {test_RMSE:.3f}'
                        f' R2 {test_R2:.3f}')
            logger.info(f'Max: '
                        f' MAE {test_MAE_max:.3f}'
                        f' MSE {test_MSE_max:.3f}'
                        f' RMSE {test_RMSE_max:.3f}'
                        f' R2 {test_R2_max:.3f}')
    return test_Loss, test_MAE, test_MSE, test_RMSE, test_R2

def loop(epoch, model, dataloader, mode="valid", optimizer=None,scheduler = None):
    loss_fn = torch.nn.SmoothL1Loss() 

    total_loss = AverageMeter()
    V_MAE = AverageMeter()
    V_MSE = AverageMeter()
    V_RMSE = AverageMeter()
    V_R2 = AverageMeter()

    N = len(dataloader)
    for step, batch in tqdm(enumerate(dataloader), disable=disable_tqdm, total=N):

        y_data = []
        y_pred = []
        y_data = np.array(y_data)
        y_pred = np.array(y_pred)

        if optimizer: 
            optimizer.zero_grad()
    
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        predictions = model(h_V, batch.edge_index, h_E, seq=batch.seq,batch33 = batch.batch,feature_molecule=batch.feature_molecule,picdata = batch.picdata)
        solu = batch.solu
        predictions = torch.squeeze(predictions)
        
        solu_list = solu.cpu().detach().numpy()
        predictions_list = predictions.cpu().detach().numpy()
        y_data = np.append(y_data,solu_list)
        y_pred = np.append(y_pred, predictions_list)
        loss_value = loss_fn(predictions, solu)
        total_loss.update(loss_value,solu.shape[0])

        if optimizer:
            loss_value.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

        V_MAE.update(mean_absolute_error(y_data, y_pred))
        V_MSE.update(mean_squared_error(y_data, y_pred))
        V_RMSE.update(sqrt(mean_squared_error(y_data, y_pred)))
        V_R2.update(r2_score(y_data, y_pred))
        if mode == "train":
            print(f'Epoch: [{epoch + 1}][{step}/{N}]'
                        f'Loss {total_loss.val:.3f} ({total_loss.avg:.3f}) '
                        f' MAE {V_MAE.val:.3f} ({V_MAE.avg:.3f}) '
                        f' MSE {V_MSE.val:.3f} ({V_MSE.avg:.3f}) '
                        f' RMSE {V_RMSE.val:.3f} ({V_RMSE.avg:.3f}) '
                        f' R2 {V_R2.val:.3f} ({V_R2.avg:.3f}) '
                        )

    return total_loss.avg, V_MAE.avg, V_MSE.avg, V_RMSE.avg, V_R2.avg
    
if __name__== "__main__":
    main()
