import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils import to_undirected

from core.data_utils.peptides_functional import PeptidesFunctionalDataset
from core.data_utils.peptides_structural import PeptidesStructuralDataset
from core.data_utils.coco import COCOSuperpixels
from core.data_utils.voc import VOCSuperpixels
from core.transform import PositionalEncodingTransform, SuperPixelPositionalEncodingTransform, RandomPartitionTransform

from core.config import cfg, update_cfg
import numpy as np

from functools import partial
import time
from torchvision import transforms

def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)

def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data 
    '''
    seq = data.y

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data

def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim = 0)
    edge_attr_ast_inverse = torch.cat([torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim = 1)

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1,) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)


    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))


    data.edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim = 1)
    data.edge_attr = torch.cat([edge_attr_ast,   edge_attr_ast_inverse, edge_attr_nextoken,  edge_attr_nextoken_inverse], dim = 0)

    return data

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

def create_dataset(cfg):
    pre_transform = PositionalEncodingTransform(
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)
    sp_pre_transform = PositionalEncodingTransform( 
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    if cfg.dataset == 'coco':
        root = 'dataset/coco'
        train_dataset = COCOSuperpixels(
            root, split='train', slic_compactness=30, pre_transform=sp_pre_transform)
        val_dataset = COCOSuperpixels(
            root, split='val', slic_compactness=30, pre_transform=sp_pre_transform)
        test_dataset = COCOSuperpixels(
            root, split='test', slic_compactness=30, pre_transform=sp_pre_transform)

    elif cfg.dataset == 'voc':
        root = 'dataset/voc'
        train_dataset = VOCSuperpixels(
            root, split='train', slic_compactness=30, pre_transform=sp_pre_transform)
        val_dataset = VOCSuperpixels(
            root, split='val', slic_compactness=30, pre_transform=sp_pre_transform)
        test_dataset = VOCSuperpixels(
            root, split='test', slic_compactness=30, pre_transform=sp_pre_transform)

    elif cfg.dataset == 'peptides-func':
        dataset = PeptidesFunctionalDataset(
            root='dataset', pre_transform=pre_transform)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                           ], dataset[split_idx['val']], dataset[split_idx['test']]

    elif cfg.dataset == 'peptides-struct':
        dataset = PeptidesStructuralDataset(
            root='dataset', pre_transform=pre_transform)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                           ], dataset[split_idx['val']], dataset[split_idx['test']]
    
    elif 'ogbg' in cfg.dataset:
        if cfg.dataset == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(name = cfg.dataset, transform = add_zeros, pre_transform=pre_transform)
        elif cfg.dataset == 'ogbg-code2':
            dataset = PygGraphPropPredDataset(name = cfg.dataset)
            split_idx = dataset.get_idx_split()
            max_seq_len = 5
            num_vocab = 5000
            vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)
            dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])
        else:
            dataset = PygGraphPropPredDataset(name = cfg.dataset)
        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]

    else:
        print("Dataset not supported.")
        exit(1)

    torch.set_num_threads(cfg.num_workers)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("Generating data")

    cfg.merge_from_file('train/configs/molhiv.yaml')
    cfg = update_cfg(cfg)
    cfg.metis.n_patches = 0
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    if cfg.dataset == 'CSL' or cfg.dataset == 'exp-classify':
        print('------------Dataset--------------')
        calculate_stats(train_dataset)
        print('------------------------------')
    else:
        print('------------Train--------------')
        calculate_stats(train_dataset)
        print('------------Validation--------------')
        calculate_stats(val_dataset)
        print('------------Test--------------')
        calculate_stats(test_dataset)
        print('------------------------------')
