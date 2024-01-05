from core.model import *


def create_model(cfg):
    if 'ogb' in cfg.dataset:
        if cfg.dataset == 'ogbg-ppa':
            nfeat_node = 1
            nfeat_edge = 7
        elif cfg.dataset == 'ogbg-code2':
            nfeat_node = None
            nfeat_edge = 2
        elif cfg.dataset == 'ogbg-molpcba' or cfg.dataset == 'ogbg-molhiv':
            nfeat_node = 3
            nfeat_edge = 9

        if "ogbg-mol" in cfg.dataset:
            node_type = 'Atom'
            edge_type = 'Bond'
        else:    
            node_type = 'Discrete'
            edge_type = 'Linear'
        
        if cfg.dataset == "ogbg-molpcba":
            nout = 128
        elif cfg.dataset == "ogbg-code2":
            nout = 5002
        elif cfg.dataset == "ogbg-ppa":
            nout = 37
        else:
            nout = 1

        out_type = "cls"

        if cfg.dataset == "ogbg-code2":
            return GraphHierAttnCode(nfeat_node=nfeat_node,
                                nfeat_edge=nfeat_edge,
                                nhid=cfg.model.hidden_size,
                                nout=nout,
                                node_type=node_type,
                                edge_type=edge_type,
                                rw_dim=cfg.pos_enc.rw_dim,
                                lap_dim=cfg.pos_enc.lap_dim,
                                dropout=cfg.train.dropout,
                                n_patches=cfg.metis.n_patches,
                                out_type=out_type)
        else:
            return GraphHierAttn(nfeat_node=nfeat_node,
                                nfeat_edge=nfeat_edge,
                                nhid=cfg.model.hidden_size,
                                nout=nout,
                                node_type=node_type,
                                edge_type=edge_type,
                                rw_dim=cfg.pos_enc.rw_dim,
                                lap_dim=cfg.pos_enc.lap_dim,
                                dropout=cfg.train.dropout,
                                n_patches=cfg.metis.n_patches,
                                out_type=out_type)