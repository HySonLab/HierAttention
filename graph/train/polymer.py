from sklearn import cluster
import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model
import shutil
import os
from datetime import date
from train.get_args import get_args

def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    N = 0
    for i, data in enumerate(train_loader):
        # print(i)
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        optimizer.zero_grad()
        out = model(data)
        loss = (out[mask].squeeze() - data.y[mask]).abs().mean() 
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    train_loss = total_loss / N
    train_perf = train_loss
    return train_perf, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    N = 0
    for data in loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        out = model(data)
        loss = (out[mask].squeeze() - data.y[mask]).abs().mean()
        # loss = ((out[mask].squeeze() - data.y[mask])**2).mean()
        total_loss += loss.item()*data.num_graphs
        N += data.num_graphs

    test_loss = total_loss/N
    test_perf = -test_loss
    return test_perf, test_loss


if __name__ == '__main__':
    args = get_args()

    cfg.merge_from_file(args.config)
    cfg = update_cfg(cfg)
    cfg.debug = args.debug
    if cfg.debug:
        print("-------DEBUG MODE-------")

    data_name = cfg.dataset
    log_folder_name = cfg.expname + '-' + str(date.today())
    os.makedirs(f'logs/{data_name}/{log_folder_name}', exist_ok=True)
    shutil.copyfile(args.config, f'logs/{data_name}/{log_folder_name}/config.yaml')
    
    run(cfg, create_dataset, create_model, train, test, evaluator=None)
