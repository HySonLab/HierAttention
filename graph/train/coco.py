import torch
import numpy as np
from core.config import cfg, update_cfg
from core.train_helper import run
from core.get_data import create_dataset
from core.get_model import create_model
from sklearn.metrics import f1_score
import shutil
import os
from datetime import date
from train.get_args import get_args
import torchmetrics
import torch.nn.functional as F

def eval_ap(y_true, y_pred):
    '''
        compute F1 score
    '''
    f1 = torchmetrics.F1Score(task = "multiclass", num_classes =81, average = "macro")
    y_true = y_true
    y_pred = y_pred.argmax(-1)


    return f1(y_pred, y_true)

def weighted_cross_entropy(pred, true):
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    torch.use_deterministic_algorithms(False)
    label_count = torch.bincount(true)
    torch.use_deterministic_algorithms(True)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred

def train(train_loader, model, optimizer, evaluator, device, sharp):
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    criterion = weighted_cross_entropy
    for data in train_loader:
        if model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.float)
        out, _ = model(data)
        y_preds.append(out.detach().cpu())
        y_trues.append(y.detach().cpu())
        loss = criterion(out[mask], y[mask].long())[0]
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    train_perf = eval_ap(y_true=y_trues, y_pred=y_preds)
    train_loss = total_loss / N
    return train_perf, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    total_loss = 0
    N = 0
    y_preds, y_trues = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(device)
        mask = ~torch.isnan(data.y)
        y = data.y.to(torch.float)
        out, _ = model(data)
        y_preds.append(out.detach().cpu())
        y_trues.append(y.detach().cpu())
        loss = criterion(out[mask], y[mask].long())
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)

    test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)
    test_loss = total_loss/N
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