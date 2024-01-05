import torch
import numpy as np
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model
from sklearn.metrics import average_precision_score
import shutil
import os
from datetime import date
from train.get_args import get_args
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from tqdm import tqdm
import random

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(train_loader, model, optimizer, evaluator, device, sharp):
    model.train()

    total_loss = 0
    N = 0
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, link_loss = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,)) + link_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            N += batch.num_graphs
            
            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    train_perf = list(evaluator.eval(input_dict).values())[0]
    train_loss = total_loss / N
    return train_perf, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    N = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(batch)

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            
            total_loss += loss.item() * batch.num_graphs
            N += batch.num_graphs

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return list(evaluator.eval(input_dict).values())[0], total_loss / N


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
    
    dataset = PygGraphPropPredDataset(name = cfg.dataset)
    evaluator = Evaluator(cfg.dataset)
    num_tasks = dataset.num_tasks
    task_type = dataset.task_type
    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator)