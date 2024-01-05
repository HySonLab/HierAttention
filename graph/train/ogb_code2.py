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

def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab

def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''


    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1, as_tuple=False) # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)] # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))

def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data 
    '''
    seq = data.y

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data

def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)


arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)

def train(train_loader, model, optimizer, evaluator, device, sharp):
    arr_to_seq, evaluator = evaluator
    model.train()

    total_loss = 0
    N = 0
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list, link_loss = model(batch)
            optimizer.zero_grad()

            loss = 0
            mat = []
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:,i])
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1).detach().cpu())
            loss = loss / len(pred_list)
            loss += link_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            N += batch.num_graphs

            mat = torch.cat(mat, dim = 1)
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [batch.y[i] for i in range(len(batch.y))]
            # seq_ref_list.extend(seq_ref)
            # seq_pred_list.extend(seq_pred)
        
    # input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    # train_perf = evaluator.eval(input_dict)
    train_loss = total_loss / N
    return -1, train_loss


@torch.no_grad()
def test(loader, model, evaluator, device):
    arr_to_seq, evaluator = evaluator
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

            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            total_loss += loss.item() * batch.num_graphs
            N += batch.num_graphs

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['rocauc'], total_loss / N


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
    split_idx = dataset.get_idx_split()
    max_seq_len = 5
    num_vocab = 5000
    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
    arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
    run(cfg, create_dataset, create_model, train, test, evaluator=[arr_to_seq, evaluator])