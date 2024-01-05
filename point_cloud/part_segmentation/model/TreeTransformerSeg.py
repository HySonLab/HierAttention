import torch
import torch.nn as nn

from torch_scatter import segment_coo, scatter_softmax, scatter_max, scatter_sum, scatter_mean

from model.lib import query_ball_point, query_knn_point

class TreeAttention(nn.Module):
    def __init__(self, scale, heads, dim):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)

        self.pe_mlp = nn.Sequential(
                                nn.Linear(3, 3), 
                                nn.BatchNorm1d(3), 
                                nn.ReLU(inplace=True), 
                                nn.Linear(3, dim // heads)
                            )

    def forward(self, ppQKVm) -> torch.Tensor:
        pq, pk, Q, K, V, m = ppQKVm

        queries, keys = m[:, 0], m[:, 1]   
        queries, keys = queries.type_as(m).long(), keys.type_as(m).long()

        q = Q[queries]
        k = K[keys]
        v = V[keys]
        pe = pq[queries].type_as(q) - pk[keys].type_as(k)
        
        q = q.view(-1, self.heads, q.shape[1]//self.heads) 
        k = k.view(-1, self.heads, k.shape[1]//self.heads)
        v = v.view(-1, self.heads, v.shape[1]//self.heads) 
        pe = self.pe_mlp(pe)

        count = torch.ones(k.shape[0], 1).type_as(Q)
        count = segment_coo(count, queries, reduce="sum", dim_size=Q.shape[0]).view(Q.shape[0], -1)
        count = count[queries]

        w = (k * q + pe.unsqueeze(1))/(self.scale*count.unsqueeze(-1) + 1e-8) 

        w = w.sum(-1, keepdims=True)
        a = scatter_softmax(w, queries.unsqueeze(-1), dim=0)
        v = (v + pe.unsqueeze(1)) * a
        out = segment_coo(v, queries, reduce="sum", dim_size=Q.shape[0]).view(Q.shape[0], -1)

        return out
    
class TreeAttentionSiblings(nn.Module):
    def __init__(self, scale, heads, dim, depth):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.k = 16
        self.radius = 0.2 * depth

        self.pe_mlp = nn.Sequential(
                                nn.Linear(3, 3), 
                                nn.BatchNorm1d(3), 
                                nn.ReLU(inplace=True), 
                                nn.Linear(3, dim // heads)
                            )

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pQKVb) -> torch.Tensor:

        p, Q, K, V, b = pQKVb

        q = []
        k = []
        v = []
        pq = []
        pk = []
        for i in range(b[-1].int().item() + 1):
            p_i = p[b == i].unsqueeze(0)
            idx = query_knn_point(self.k, p_i, p_i).squeeze()
            if idx.shape[1] < self.k:
                idx_extend = idx[:, 0].unsqueeze(-1).expand(-1, self.k - idx.shape[1])
                idx = torch.cat([idx, idx_extend], dim = -1)

            q.append(Q[b == i].unsqueeze(1).expand(-1, self.k, -1).flatten(0,1))
            k.append(K[b == i][idx].flatten(0,1))
            v.append(V[b == i][idx].flatten(0,1))

            pq.append(p[b == i].unsqueeze(1).expand(-1, self.k, -1).flatten(0,1))
            pk.append(p[b == i][idx].flatten(0,1))

        q = torch.cat(q, dim = 0)
        k = torch.cat(k, dim = 0)
        v = torch.cat(v, dim = 0)
        pq = torch.cat(pq, dim = 0)
        pk = torch.cat(pk, dim = 0) 
        
        pe = pq - pk
        
        q = q.view(-1, self.heads, q.shape[1]//self.heads) 
        k = k.view(-1, self.heads, k.shape[1]//self.heads)
        v = v.view(-1, self.heads, v.shape[1]//self.heads) 
        pe = self.pe_mlp(pe)


        w = (k * q + pe.unsqueeze(1))/(self.scale*self.k + 1e-8) 

        w = w.sum(-1, keepdims=True)
        w = w.view(-1, self.k, w.shape[-2], w.shape[-1])

        
        a = self.softmax(w)

        pe = pe.view(-1, self.k, pe.shape[-1])
        v = v.view(-1, self.k, v.shape[-2], v.shape[-1])
        v = (v + pe.unsqueeze(2)) * a    

        
        out = v.sum(1).view(Q.shape[0], -1)

        return out

class TreeLayer(nn.Module):
    def __init__(self, in_planes, out_planes, heads, depth, expand):
        super(TreeLayer, self).__init__()
        self.heads = heads
        self.scale = out_planes**0.5
        self.depth = depth

        self.linearQ = []
        self.linearK = []
        self.linearV = []
        for i in range(self.depth):
            self.linearQ.append(nn.Linear(in_planes*expand**i, out_planes*expand**i, bias=False))
            self.linearK.append(nn.Linear(in_planes*expand**i, out_planes*expand**i, bias=False))
            self.linearV.append(nn.Linear(in_planes*expand**i, out_planes*expand**i, bias=False))
        self.linearQ = nn.ModuleList(self.linearQ)
        self.linearK = nn.ModuleList(self.linearK)
        self.linearV = nn.ModuleList(self.linearV)

        self.expands_K = []
        self.shrinks_K = []
        self.expands_V = []
        self.shrinks_V = []
        for i in range(self.depth):
            if i != 0:
                self.expands_K.append(nn.Linear(out_planes*expand**(i-1), out_planes*expand**i, bias=False))
                self.expands_V.append(nn.Linear(out_planes*expand**(i-1), out_planes*expand**i, bias=False))
            if i != self.depth - 1:
                self.shrinks_K.append(nn.Linear(out_planes*expand**(i+1), out_planes*expand**i, bias=False))
                self.shrinks_V.append(nn.Linear(out_planes*expand**(i+1), out_planes*expand**i, bias=False))
        self.expands_K = nn.ModuleList(self.expands_K)
        self.shrinks_K = nn.ModuleList(self.shrinks_K)
        self.expands_V = nn.ModuleList(self.expands_V)
        self.shrinks_V = nn.ModuleList(self.shrinks_V)

        self.parents_attns = []
        self.children_attns = []
        self.siblings_attns = []
        for i in range(self.depth):
            self.parents_attns.append(TreeAttention(out_planes**0.5, heads, out_planes*expand**i))
            self.children_attns.append(TreeAttention(out_planes**0.5, heads, out_planes*expand**i))
            self.siblings_attns.append(TreeAttentionSiblings(out_planes**0.5, heads, out_planes*expand**i, i+1))
        self.parents_attns = nn.ModuleList(self.parents_attns)
        self.children_attns = nn.ModuleList(self.children_attns)
        self.siblings_attns = nn.ModuleList(self.siblings_attns)

        self.weight = nn.Parameter(torch.empty(3).normal_(mean=0,std=1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, pxmb):
        p_c, x_c, m_c, b_c = pxmb  # (b*n, 3), (b*n, c), (b), (m, 5)

        Q = []
        K = []
        V = []
        for i, x in enumerate(x_c):    
            Q.append(self.linearQ[i](x))
            K.append(self.linearK[i](x))
            V.append(self.linearV[i](x))

        parents_mapping = m_c[:, [1,0,3,2]]
        children_mapping = m_c
        out_c = []
        for i, x in enumerate(x_c): 
            parents_m = parents_mapping[parents_mapping[:, 2] == self.depth - 1 - i]
            parents_m = parents_m[torch.argsort(parents_m[:, 0])]
            children_m =  children_mapping[children_mapping[:, 2] == self.depth - 1 - i]
            children_m = children_m[torch.argsort(children_m[:, 0])]

            parents = self.parents_attns[i]([p_c[i], p_c[i+1], Q[i], self.shrinks_K[i](K[i+1]), self.shrinks_V[i](V[i+1]), parents_m]) if i != self.depth - 1 else 0
            siblings = self.siblings_attns[i]([p_c[i], Q[i], K[i], V[i], b_c[i]]) if i != self.depth - 1 else 0
            children = self.children_attns[i]([p_c[i], p_c[i-1], Q[i], self.expands_K[i-1](K[i-1]), self.expands_V[i-1](V[i-1]), children_m]) if i != 0 else 0

            weight = self.softmax(self.weight)
      
            out = weight[0] * parents + weight[1] * children + weight[2] * siblings
            out_c.append(out)
        
        return out_c

# MLP -> Attention -> MLP
class TreeBlock(nn.Module): 
    def __init__(self, in_planes, out_planes, heads, depth, expand):
        super(TreeBlock, self).__init__()
        self.heads = heads
        self.depth = depth

        self.mlp1s = []
        for i in range(self.depth):
            self.mlp1s.append(
                nn.Sequential(
                    nn.Linear(in_planes*expand**i, out_planes*expand**i, bias=False),
                    nn.BatchNorm1d(out_planes*expand**i),
                    nn.ReLU(inplace=True),
                )
            )
        self.mlp1s = nn.ModuleList(self.mlp1s)

        self.attn2a = TreeLayer(out_planes, out_planes, heads, depth, expand)
        self.attn2bs = []
        for i in range(self.depth):
            self.attn2bs.append( 
                nn.Sequential(
                    nn.BatchNorm1d(out_planes*expand**i),
                    nn.ReLU(inplace=True),
                )
            )
        self.attn2bs = nn.ModuleList(self.attn2bs)

        self.mlp3as = []
        for i in range(self.depth):
            self.mlp3as.append(
                nn.Sequential(
                    nn.Linear(out_planes*expand**i, out_planes*expand**i, bias=False),
                    nn.BatchNorm1d(out_planes*expand**i),
                )
            )
        self.mlp3as = nn.ModuleList(self.mlp3as)
        self.mlp3b = nn.ReLU(inplace=True)

    def forward(self, pxmb):
        p_c, x_c, m_c, b_c = pxmb

        identity = []
        for x in x_c:
            identity.append(x)

        for i, x in enumerate(x_c):
            x_c[i] = self.mlp1s[i](x)
        
        x_c = self.attn2a([p_c, x_c, m_c, b_c])

        for i, x in enumerate(x_c):
            x_c[i] = self.attn2bs[i](x)

        for i, x in enumerate(x_c):
            x = self.mlp3as[i](x)
            x = x + identity[i]
            x = self.mlp3b(x)

        return x_c

class Init(nn.Module):
    def __init__(self, in_channel, dim, depth, expand):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.projections = []
        for i in range(0, self.depth):
            if i == 0:
                self.projections.append(
                    nn.Sequential(
                        nn.Linear(in_channel, self.dim*expand**i),
                        nn.BatchNorm1d(self.dim*expand**i),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.projections.append(
                    nn.Sequential(
                        nn.Linear(self.dim*expand**(i-1), self.dim*expand**i),
                        nn.BatchNorm1d(self.dim*expand**i),
                        nn.ReLU(inplace=True),
                    )
                )
        self.projections = nn.ModuleList(self.projections)
        
    def forward(self, pxbim):
        p, x, b, idx, m = pxbim
        p_c = []
        x_c = []
        b_c = []

        x_proj = x

        for i, ii in enumerate(idx):
            x_proj = self.projections[i](x_proj[ii.long()])
            x_c.append(x_proj)
            b_c.append(b[ii.long()])
            p_c.append(p[ii.long()])

            # import pdb; pdb.set_trace()
            m_proj = m[m[:, 2] == self.depth - i - 2][:, 0]
            

            if len(m_proj) != x_proj.shape[0]:
                unique, count = torch.unique(m[m[:, 2] == self.depth - i - 2][:, 1], return_counts=True)
                duplicate = unique[count > 1]
                for d in duplicate:
                    idxxx = torch.where(m[m[:, 2] == self.depth - i - 2][:, 1] == d)[0][1:]
                    m_proj[idxxx] = -1
                m_proj = m_proj[m_proj != -1]
            
            
            if i != len(idx) - 1:
                temp, _ = scatter_max(x_proj, m_proj.long().unsqueeze(-1), dim = 0)
                x_proj = temp
            

        return p_c, x_c, b_c

class TreeTransformerSeg(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n = args.n_point
        self.c = args.c
        self.dim = args.dim
        self.heads = args.heads
        self.depth = args.depth + 1
        self.expand = 2
        self.layers = args.layers

        self._init = Init(self.c, self.dim, self.depth, self.expand)

        self.enc = []
        for i in range(self.layers):
            if i == 0:
                block = TreeBlock(self.dim, self.dim, self.heads, self.depth, self.expand)
                self.enc.append(block)
                self.shared_weight = block.parameters
            else:
                block = TreeBlock(self.dim, self.dim, self.heads, self.depth, self.expand)
                block.parameters = self.shared_weight
                self.enc.append(block)
        self.enc = torch.nn.ModuleList(self.enc)

        self.cls = nn.Sequential(
                nn.Linear(self.dim + args.n_class, 512), 
                nn.LayerNorm(512), 
                nn.ReLU(inplace=True), 
                nn.Linear(512, args.n_part)
            )

    
    def _pooling(self, x_c, idx, method = "average"):
        return x_c[0]

    def _index(self, m, length):
        idx = []
        sequence = torch.cat([m[:, [0, 2]], m[:, [1, 3]]], dim = 0).long()

        dic = torch.zeros(length).type_as(m).long()
        for d in range(self.depth):
            s = sequence[:, 0][torch.where((sequence[:, 1] == d))]
            i, inverse = torch.unique(s, sorted = True, return_inverse = True)
            idx.append(i)
            dic[s] = inverse.type_as(dic)

        m[:, 0] = dic[m[:, 0].long()]
        m[:, 1] = dic[m[:, 1].long()]
        
        idx.reverse()
        m_c = m

        return idx, m_c

    def forward(self, pxomc):
        p, x, o, m, c = pxomc  # (n, 3), (n, c), (b)
        length = x.shape[0]

        m_org = m.clone()

        b = torch.zeros(p.shape[0])
        begin = 0
        for i, end in enumerate(o):
            b[begin:end] = i
            begin = end

        idx, m_c = self._index(m, length)

        # B, N, D -> 64, 128, 256
        p_c, x_c, b_c = self._init([p, x, b, idx, m_org])

        for l in range(self.layers):
            x_c = self.enc[l]([p_c, x_c, m_c, b_c])
        
        x = self._pooling(x_c, idx)
        c = c.expand(-1, x.shape[0]//c.shape[0], -1)
        c = c.reshape(-1, c.shape[-1])

        x = torch.cat([x,c], dim = -1)
        x = self.cls(x)

        return x