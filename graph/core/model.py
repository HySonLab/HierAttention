import torch
import torch.nn as nn
import torch_geometric
from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from torch_geometric.utils import to_dense_batch, to_dense_adj
from .graph_gps import GPSLayer
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import pandas as pd
import os

def dense_to_sparse(adj, attr):
    edge_index = adj.nonzero().t()
    edge_attr = attr[edge_index[0], edge_index[1], edge_index[2]]
    edge_attr_count = adj[edge_index[0], edge_index[1], edge_index[2]]
    row = edge_index[1] + adj.size(-2) * edge_index[0]
    col = edge_index[2] + adj.size(-1) * edge_index[0]
    return torch.stack([row, col], dim=0), edge_attr / edge_attr_count.unsqueeze(-1)

class GraphHierAttn(nn.Module):
    def __init__(self,
                 nfeat_node, 
                 nfeat_edge,
                 nhid, 
                 nout,
                 node_type, 
                 edge_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 n_patches=32,
                 out_type=None):

        super().__init__()
        self.nout = nout
        self.edge_type = edge_type
        self.dropout = dropout
        self.n_patches = n_patches

        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.out_type = out_type

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gps_leaf_mp = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='None', num_heads=4)
        self.gps_cluster = GPSLayer(dim_h=nhid, dim_out=self.n_patches, local_gnn_type='None', global_model_type='None', num_heads=4, ffw_sum=False)
        self.gps_leaf_attn = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='Transformer', num_heads=4, attn_dropout=0.5, sorted=True)
        self.gps_sub = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='None', num_heads=4, attn_dropout=0.5) 
        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        # Initialization
        x = self.input_encoder(data.x.squeeze()) # Get input
        if self.use_rw: # Get random walk encoder
            data.rw_pos_enc = self.rw_encoder(data.rw_pos_enc)
            data.x = x + data.rw_pos_enc
        if self.use_lap: # Get laplacian encoder
            data.lap_pos_enc = self.lap_encoder(data.lap_pos_enc)
            data.x = x + data.lap_pos_enc
        else:
            data.x = x

        edge_index= data.edge_index # Get edge index
        adj = to_dense_adj(edge_index, data.batch) # Get dense edge index

        edge_attr= data.edge_attr # Get edge atribute
        edge_attr_ori= data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr) # Encode edge attribute
        data.edge_attr = edge_attr # Assign encoded edge attribute to data

        # Clone data
        ori_data = data.clone()

        # import pdb; pdb.set_trace()

        # Learning cluster
        data = self.gps_leaf_mp(data) # Extract local feature
        w = self.gps_cluster(data).x
        x_dense, mask_x = to_dense_batch(data.x, data.batch)
        w_dense, mask_w = to_dense_batch(w, data.batch) # Get weight dense

        # Calculate loss for cluster
        w_dense = F.gumbel_softmax(w_dense, tau = 1.0, hard = True, dim = -1) # Gumbel softmax
        link_loss = adj - (torch.matmul(w_dense, w_dense.transpose(1, 2)) - torch.eye(w_dense.shape[1]).type_as(w)) # Calculate loss
        link_loss = torch.norm(link_loss, p=2)/(link_loss.shape[-1]**2) # Normalize loss

        # Get index
        index = w_dense.nonzero()[mask_x.flatten()] # Get index 
        index = index[:, 0]*self.n_patches + index[:, 2] # Convert index to sparse
        data.batch = index # Assign index to data

        # Create fake node for attention
        x = ori_data.x # Get x from original data
        len_x = x.shape[0] # Get len x
        a = torch.ones(self.n_patches*ori_data.num_graphs).bool() # Create mask for all nodes
        u = torch.unique(data.batch) # Check node that are not empty
        a[u] = False # Set empty node = False
        cat_batch = torch.where(a)[0].type_as(data.batch) # Create fake batch
        cat_x = torch.ones(len(cat_batch), x.shape[-1]).type_as(x) # Create fake nodes

        # Create edge for subgraph
        edge_subgraph_dense = torch.einsum('bmc, bmn, bnd -> bcd', w_dense, adj, w_dense)
        E = to_dense_adj(ori_data.edge_index, ori_data.batch, edge_attr = edge_attr_ori.float())
        edge_mask = torch.eye(edge_subgraph_dense.shape[1], edge_subgraph_dense.shape[2])
        edge_attr_subgraph_dense = torch.einsum('bmc, bmnk, bnd -> bcdk', w_dense, E, w_dense)
        edge_subgraph_dense = edge_subgraph_dense.masked_fill_(edge_mask.type_as(edge_subgraph_dense).bool(), 0)
        edge_index_subgraph, edge_attr_subgraph = dense_to_sparse(edge_subgraph_dense, edge_attr_subgraph_dense)

        # Encode subgraph attribute
        if self.edge_type != 'Linear':
            edge_attr_subgraph = edge_attr_subgraph.long()
        edge_attr_subgraph = self.edge_encoder(edge_attr_subgraph)

        # Create input for gps on node level
        batch_node_attn = Batch() # Create batch class
        batch_node_attn.batch = torch.cat([data.batch, cat_batch], dim = 0) # Add batch
        batch_node_attn.edge_index = edge_index # Add edge index
        batch_node_attn.edge_attr = edge_attr # Add edge attribute


        # Create input for gps on subgraph level
        mask_s = (torch.arange(ori_data.num_graphs).repeat_interleave(self.n_patches)).type_as(ori_data.batch) # Create batch subgraph
        batch_subgraph = Batch() # Create batch class
        batch_subgraph.batch = mask_s # Add batch
        batch_subgraph.edge_index = edge_index_subgraph # Add edge index
        batch_subgraph.edge_attr = edge_attr_subgraph # Add edge attribute

        # Add new node
        batch_node_attn.x = torch.cat([x, cat_x], dim = 0) # Add node

        # Attention on node
        batch_node_attn = self.gps_leaf_attn(batch_node_attn) # Apply gcn + attention layer
        x_out = batch_node_attn.x[:len_x] # Remove fake nodes

        # Calculate subgraph feature
        s = global_add_pool(x_out, index, size=ori_data.num_graphs*self.n_patches)

        # GPS on subgraph
        batch_subgraph.x = s # Add node
        batch_subgraph = self.gps_sub(batch_subgraph) # Apply attention layer
        s = batch_subgraph.x # Get subgraph after self-attention

        x = global_add_pool(s, mask_s)
        x = self.output_decoder(x)
        return x, link_loss

class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.

        Output:
            emb_dim-dimensional vector

    '''
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)


    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:,0]) + self.attribute_encoder(x[:,1]) + self.depth_encoder(depth)

class GraphHierAttnCode(nn.Module):
    def __init__(self,
                 nfeat_node, 
                 nfeat_edge,
                 nhid, 
                 nout,
                 node_type, 
                 edge_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 n_patches=32,
                 out_type=None):

        super().__init__()
        self.nout = nout
        self.edge_type = edge_type
        self.dropout = dropout
        self.n_patches = n_patches

        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.out_type = out_type

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        nodetypes_mapping = pd.read_csv(os.path.join('dataset/ogbg_code2', 'mapping', 'typeidx2type.csv.gz'))
        nodeattributes_mapping = pd.read_csv(os.path.join('dataset/ogbg_code2', 'mapping', 'attridx2attr.csv.gz'))


        self.input_encoder = ASTNodeEncoder(nhid, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gps_leaf_mp = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='None', num_heads=4)
        self.gps_cluster = GPSLayer(dim_h=nhid, dim_out=self.n_patches, local_gnn_type='None', global_model_type='None', num_heads=4, ffw_sum=False)
        self.gps_leaf_attn = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='Transformer', num_heads=4, attn_dropout=0.5, sorted=True)
        self.gps_sub = GPSLayer(dim_h=nhid, local_gnn_type='CustomGatedGCN', global_model_type='None', num_heads=4, attn_dropout=0.5) 
        
        max_seq_len = 5
        self.output_decoder_list = []
        for i in range(max_seq_len):
            self.output_decoder_list.append(MLP(nhid, nout, nlayer=2, with_final_activation=False))
        self.output_decoder_list = nn.ModuleList(self.output_decoder_list)
    def forward(self, data):
        # Initialization
        x = self.input_encoder(data.x.squeeze(), data.node_depth.view(-1,)) # Get input
        if self.use_rw: # Get random walk encoder
            data.rw_pos_enc = self.rw_encoder(data.rw_pos_enc)
            data.x = x + data.rw_pos_enc
        if self.use_lap: # Get laplacian encoder
            data.lap_pos_enc = self.lap_encoder(data.lap_pos_enc)
            data.x = x + data.lap_pos_enc
        else:
            data.x = x

        edge_index= data.edge_index # Get edge index
        adj = to_dense_adj(edge_index, data.batch) # Get dense edge index

        edge_attr= data.edge_attr # Get edge atribute
        edge_attr_ori= data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr) # Encode edge attribute
        data.edge_attr = edge_attr # Assign encoded edge attribute to data

        # Clone data
        ori_data = data.clone()

        # import pdb; pdb.set_trace()

        # Learning cluster
        data = self.gps_leaf_mp(data) # Extract local feature
        w = self.gps_cluster(data).x
        x_dense, mask_x = to_dense_batch(data.x, data.batch)
        w_dense, mask_w = to_dense_batch(w, data.batch) # Get weight dense

        # Calculate loss for cluster
        w_dense = F.gumbel_softmax(w_dense, tau = 1.0, hard = True, dim = -1) # Gumbel softmax
        link_loss = adj - (torch.matmul(w_dense, w_dense.transpose(1, 2)) - torch.eye(w_dense.shape[1]).type_as(w)) # Calculate loss
        link_loss = torch.norm(link_loss, p=2)/(link_loss.shape[-1]**2) # Normalize loss

        # Get index
        index = w_dense.nonzero()[mask_x.flatten()] # Get index 
        index = index[:, 0]*self.n_patches + index[:, 2] # Convert index to sparse
        data.batch = index # Assign index to data

        # Create fake node for attention
        x = ori_data.x # Get x from original data
        len_x = x.shape[0] # Get len x
        a = torch.ones(self.n_patches*ori_data.num_graphs).bool() # Create mask for all nodes
        u = torch.unique(data.batch) # Check node that are not empty
        a[u] = False # Set empty node = False
        cat_batch = torch.where(a)[0].type_as(data.batch) # Create fake batch
        cat_x = torch.ones(len(cat_batch), x.shape[-1]).type_as(x) # Create fake nodes

        # Create edge for subgraph
        edge_subgraph_dense = torch.einsum('bmc, bmn, bnd -> bcd', w_dense, adj, w_dense)
        E = to_dense_adj(ori_data.edge_index, ori_data.batch, edge_attr = edge_attr_ori.float())
        edge_mask = torch.eye(edge_subgraph_dense.shape[1], edge_subgraph_dense.shape[2])
        edge_attr_subgraph_dense = torch.einsum('bmc, bmnk, bnd -> bcdk', w_dense, E, w_dense)
        edge_subgraph_dense = edge_subgraph_dense.masked_fill_(edge_mask.type_as(edge_subgraph_dense).bool(), 0)
        edge_index_subgraph, edge_attr_subgraph = dense_to_sparse(edge_subgraph_dense, edge_attr_subgraph_dense)

        # Encode subgraph attribute
        if self.edge_type != 'Linear':
            edge_attr_subgraph = edge_attr_subgraph.long()
        edge_attr_subgraph = self.edge_encoder(edge_attr_subgraph)

        # Create input for gps on node level
        batch_node_attn = Batch() # Create batch class
        batch_node_attn.batch = torch.cat([data.batch, cat_batch], dim = 0) # Add batch
        batch_node_attn.edge_index = edge_index # Add edge index
        batch_node_attn.edge_attr = edge_attr # Add edge attribute


        # Create input for gps on subgraph level
        mask_s = (torch.arange(ori_data.num_graphs).repeat_interleave(self.n_patches)).type_as(ori_data.batch) # Create batch subgraph
        batch_subgraph = Batch() # Create batch class
        batch_subgraph.batch = mask_s # Add batch
        batch_subgraph.edge_index = edge_index_subgraph # Add edge index
        batch_subgraph.edge_attr = edge_attr_subgraph # Add edge attribute

        # Add new node
        batch_node_attn.x = torch.cat([x, cat_x], dim = 0) # Add node

        # Attention on node
        batch_node_attn = self.gps_leaf_attn(batch_node_attn) # Apply gcn + attention layer
        x_out = batch_node_attn.x[:len_x] # Remove fake nodes

        # Calculate subgraph feature
        s = global_add_pool(x_out, index, size=ori_data.num_graphs*self.n_patches)

        # GPS on subgraph
        batch_subgraph.x = s # Add node
        batch_subgraph = self.gps_sub(batch_subgraph) # Apply attention layer
        s = batch_subgraph.x # Get subgraph after self-attention

        x = global_add_pool(s, mask_s)
        pred_list = []
        for layer in self.output_decoder_list:
            pred_list.append(layer(x))
        return pred_list, link_loss