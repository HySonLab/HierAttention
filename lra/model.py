import torch
import torch.nn as nn
import math
import os
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import *

class SequoiaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.h = config["num_head"] # Number of heads
        self.d = config["head_dim"] # Headwise dimension
        self.k = config["k"] # Division factor
        self.max_seq_len = config["max_seq_len"]

        self.seq_mode = config["seq_mode"]
        self.seq_orders = {
            "bottom-up": lambda L: list(range(L)),
            "top-down": lambda L: list(range(L))[::-1],
            "interleaved": lambda L: [val for pair in zip(list(range(L)), list(range(L))[::-1]) for val in pair],
            "complete": lambda L: list(range(L)) + list(range(L))[::-1],
            "random": lambda L: torch.randperm(L),
        }
        self.L = math.ceil(math.log(self.max_seq_len) / math.log(self.k)) + 1 # Number of levels of the tree
        self.virtual_pos_encodings = config["virtual_pos_encodings"]
        self.virtual_embeddings = config["virtual_embeddings"]
        self.probe_layer = config["probe_layer"]
        self.drop_ancestors = False
        self.drop_siblings = False
        self.drop_children = False

    def forward(self, Q, K, V, mask=None):
        b, h, n_tot, d = Q.shape
        mask = True
        assert (h == self.h) and (d == self.d) and (n_tot == self.n_tot)
        n = self.n

        L = math.ceil(math.log(n) / math.log(self.k)) + 1 # Number of levels of the tree
        L_indices_relative = [0] + [math.ceil(n / (self.k ** i)) for i in range(L)] # Number of nodes at each level of the tree
        L_indices = np.cumsum(np.array(L_indices_relative)) # Starting indices of each level of the tree (in absolute indices)
        def sdp(l, V, selector):
            queries, keys, values = Q[:, :, L_indices[l]: L_indices[l+1], :], K[:, :, selector, :], V[:, :, selector, :] # Select queries, keys and values using the selector mapping
            interactions = torch.einsum("bhnd,bhnkd->bhnk", queries, keys) / (self.d ** 0.5) # Compute scaled dot-product attention
            attention_weights = nn.functional.softmax(interactions, dim=-1)
            return torch.einsum("bhnk,bhnkd->bhnd", attention_weights, values)

        @lru_cache(maxsize=None)
        def get_selectors(l, n):
            queries, children, ancestors = range(L_indices_relative[l + 1]), None, None # No ancestor for the root node nor for leaf nodes !
            def get_ancestor(j, i): 
                if not mask:
                    rel_index = i // (self.k ** (j - l))
                    return L_indices[j] + rel_index
                else:
                    if (j + 1) / L_indices_relative[l + 2] <= i / L_indices_relative[l+1]:
                        return L_indices[l + 1] + j
                    else:
                        return 0
            def get_sibling(sibling, i): 
                rel_index = min((i // self.k) * self.k + sibling, L_indices_relative[l + 1] - 1)
                return L_indices[l] + rel_index if (not mask) or (rel_index / L_indices_relative[l + 1] <= i / L_indices_relative[l + 1]) else 0
            def get_child(child, i): 
                rel_index = min(i * self.k + child, L_indices_relative[l] - 1)
                return L_indices[l - 1] + rel_index if (not mask) or (rel_index / L_indices_relative[l] <= i / L_indices_relative[l + 1]) else 0
            if l < L - 1:
                if not mask:
                    ancestors = torch.Tensor(np.array([[get_ancestor(level, i) for level in range(l + 1, L)] for i in queries])).cuda().long()
                else:
                    ancestors = torch.Tensor(np.array([[get_ancestor(j, i) for j in range(L_indices_relative[l + 2])] for i in queries])).cuda().long()
            siblings = torch.Tensor(np.array([[get_sibling(sibling, i) for sibling in range(self.k)] for i in queries])).cuda().long()
            if l >= 1:
                children = torch.Tensor(np.array([[get_child(child, i) for child in range(self.k)] for i in queries])).cuda().long()
            return ancestors, siblings, children

        def attention(l, V):
            ancestors, siblings, children = get_selectors(l, n) # 1. Selecting the nodes at level l
            inactive = [(ancestors is None) or (self.drop_ancestors), (self.drop_siblings), (children is None) or (self.drop_children)]
            attn_ancestors = 0 if inactive[0] else sdp(l, V, ancestors)
            attn_siblings = 0 if inactive[1] else sdp(l, V, siblings) 
            attn_children = 0 if inactive[2] else sdp(l, V, children)
            mask = np.array([(-np.inf if inactive[i] else 0) for i in range(3)])
            out = (attn_ancestors + attn_siblings + attn_children) / 3
            return out

        if self.seq_mode in ["bottom-up", "top-down", "complete", "interleaved", "random"]:
            for l in self.seq_orders[self.seq_mode](L):
                V[:, :, L_indices[l]: L_indices[l + 1]] = attention(l, V)
        return V



class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]
        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.attn = SequoiaAttention(config)
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, mask, X_0=None, X_past=None, t_pred=None, X_q_above=None, X_t=None):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        with torch.cuda.amp.autocast(enabled = False):
            if self.grad_checkpointing:
                attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask)
            else:
                attn_out = self.attn(Q.float(), K.float(), V.float(), mask)
        attn_out = self.combine_heads(attn_out)

        return self.ff(attn_out)

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X



def pooling(inp, mode):
    if mode == "CLS":
        pooled = inp[:, 0, :]
    elif mode == "MEAN":
        pooled = inp.mean(dim = 1)
    else:
        raise Exception()
    return pooled

def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.causal = config["causal"]
        if self.causal == False:
            self.mlpblock = nn.Sequential(
                nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["transformer_hidden_dim"], config["num_classes"])
            )
        else:
            self.mlpblock = nn.Sequential(
                nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
                nn.ReLU(),
                nn.Linear(config["transformer_hidden_dim"], config["vocab_size"])
            )
    def forward(self, inp):
        if self.causal == False:
            seq_score = self.mlpblock(pooling(inp, self.pooling_mode))
        else:
            seq_score = self.mlpblock(inp)
        return seq_score

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        return X_token, X_pos


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config)
        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p = config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p = config["dropout_prob"])
        )

    def forward(self, X, mask, **kwargs):
        norm_out = self.norm1(X)
        att = self.mha(norm_out, mask, **kwargs)
        X = self.dropout1(att) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]
        try:
            self.tied_weights = config["layer_sharing"]
        except:
            pass

        self.embeddings = Embeddings(config)

        if self.tied_weights:
            self.transformer = Transformer(config)
        else:
            for idx in range(self.num_layers):
                os.environ['num_layer'] = str(idx)
                os.environ['num_layers'] = str(self.num_layers)
                if idx >= 1:
                    setattr(self, f"transformer_{idx}", Transformer({**config, "transformer_0": self.transformer_0}))
                else:
                    setattr(self, f"transformer_{idx}", Transformer(config))
                self.trans_idx(idx).num_layer = idx

        self.norm = nn.LayerNorm(config["transformer_dim"])
        self.causal = config["causal"]

    def trans_idx(self, idx):
        if self.tied_weights:
            name = "transformer"
        else:
            name = f"transformer_{idx}"
        return getattr(self, name).mha.attn

    def forward(self, input_ids, mask = None, t_pred=None):
        X_token, X_pos = self.embeddings(input_ids)
        X = X_token + X_pos

        b, n, d = X.shape
        k = self.trans_idx(0).k

        L = math.ceil(math.log(n) / math.log(k)) # Number of levels of the tree
        L_indices_relative = [0] + [math.ceil(n / (k ** i)) for i in range(L + 1)] # Number of nodes at each level of the tree
        L_indices = np.cumsum(np.array(L_indices_relative))
        self.n_tot = L_indices[-1]
        self.n = n
        self.n_v = self.n_tot - self.n
        for j in range(self.num_layers):
            self.trans_idx(j).n_tot = self.n_tot
            self.trans_idx(j).n = self.n
            self.trans_idx(j).n_v = self.n_v

        X_v = torch.zeros((b, self.n_v, d), device = X.device)
        X = torch.cat((X, X_v), dim = 1)

        for l in range(1, L + 1):
            queries = range(L_indices_relative[l + 1])
            children = torch.Tensor(L_indices[l - 1] + np.array([[min(i * k + child, L_indices_relative[l] - 1) for child in range(k)] for i in queries])).cuda().long()
            X[:, L_indices[l]: L_indices[l+1], :] = torch.mean(X[:, children, :], dim = 2)

        for l in range(1, L + 1):
            position = (0.5 + torch.arange(L_indices_relative[l + 1]).cuda().long()[:, np.newaxis]) * (self.n / L_indices_relative[l + 1])
            div_term = torch.exp(torch.arange(0, d, 2, device = X.device) * - (math.log(10_000.0) / d))
            pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(L_indices_relative[l + 1], -1)
            X[:, L_indices[l]: L_indices[l + 1], :] += pos_embed
            X[:, L_indices[l]: L_indices[l + 1], -l] = 0
        probe_layer = self.trans_idx(0).probe_layer

        X = self.embeddings.dropout(X)

        if self.tied_weights:
            for idx in range(self.num_layers):
                X = self.transformer(X, mask=self.causal)
        else:
            for idx in range(self.num_layers):
                X = getattr(self, f"transformer_{idx}")(X, mask=self.causal)

        if probe_layer == "root":
            X = X[:, L_indices[-2]:, :]
        elif probe_layer == "root_extended":
            X = X[:, L_indices[-3]:, :]
        elif probe_layer == "virtual":
            _, X = X[:, :n, :], X[:, n:, :] # Only get the virtual nodes
        elif probe_layer == "virtual_next":
            _, X = X[:, :n, :], X[:, n+1:, :] # Only get the virtual nodes
        elif probe_layer == "all":
            pass # Get all nodes
        else:
            X, _ = X[:, :n, :], X[:, n:, :] # Only get the true nodes (default behaviour)
        X = self.norm(X)

        return X



class ModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = True
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        self.model = Model(config)

        self.seq_classifer = ClassificationHead(config)

        self.causal = config["causal"]

    def forward(self, input_ids, mask=None, label=None):
        with torch.cuda.amp.autocast(enabled = self.enable_amp):
            if self.pooling_mode == "CLS":
                input_ids, mask = append_cls(input_ids, mask, self.vocab_size)

            token_out = self.model(input_ids, mask)
            seq_scores = self.seq_classifer(token_out) 

            if self.causal:
                seq_scores = seq_scores[:, :-1]
                label = input_ids[:, 1:]
                hard_predictions = torch.argmax(seq_scores, dim = -1)
                seq_scores = rearrange(seq_scores, 'b n c -> (b n) c')
                hard_predictions = rearrange(hard_predictions, 'b n -> (b n)')
                label = rearrange(label, 'b n -> (b n)')
            else:
                hard_predictions = torch.argmax(seq_scores, dim = -1)
            seq_loss = torch.nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (hard_predictions == label).to(torch.float32)
            seq_f1 = f1_score(label.cpu().data, hard_predictions.cpu(), average = None)
            seq_precision = precision_score(label.cpu().data, hard_predictions.cpu(), average = None)
            seq_recall = recall_score(label.cpu().data, hard_predictions.cpu(), average = None)
            metrics = {
                "seq_loss": seq_loss,
                "seq_accu": seq_accu,
                "seq_f1": seq_f1,
                "seq_precision": seq_precision,
                "seq_recall": seq_recall,
            }

        return metrics

config = {
    # MODEL PARAMETERS
    "learn_pos_emb":True,
    "tied_weights":False,
    "embedding_dim":128,
    "transformer_dim":128,
    "transformer_hidden_dim":128,
    "head_dim":128,
    "num_head":4,
    "num_layers":4,
    "dropout_prob":0.1,
    "attention_dropout":0.1,
    "pooling_mode":"MEAN",

    # PARAMETERS OF SEQUOIA ATTENTION : REPLACE WITH YOUR OWN
    "k": 8,
    "seq_mode": "bottom-up",
    "causal": False,
    "attention_grad_checkpointing":False,
    "virtual_pos_encodings": "sin",
    "virtual_embeddings": "avg",
    "probe_layer": "true",

    # DATASET PARAMETERS : num_classes is the number of output classes for classification. Vocab_size is the number of different discrete input token types (e.g. nb of words in a dictionnary)
    "num_classes": 100,
    "vocab_size": 100,
    "max_seq_len": 512,
}

model = ModelWrapper(config)
model = model.cuda()
model.train()
dummy_batch = torch.randint(0, 100, (32, 512)).cuda()
dummy_label = torch.randint(0, 100, (32,)).cuda()

model(input_ids=dummy_batch, label=dummy_label)