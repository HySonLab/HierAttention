import torch
import torch.nn as nn

class AttnLayer(nn.Module):
    def __init__(self, dim, head):
        super(AttnLayer, self).__init__()
        self.attn_leaf_parents = torch.nn.MultiheadAttention(dim, head, batch_first=True)
        self.attn_leaf_siblings = torch.nn.MultiheadAttention(dim, head, batch_first=True)

        self.attn_inner_parents = torch.nn.MultiheadAttention(dim, head, batch_first=True)
        self.attn_inner_siblings = torch.nn.MultiheadAttention(dim, head, batch_first=True)
        self.attn_inner_children = torch.nn.MultiheadAttention(dim, head, batch_first=True)


        self.attn_root_siblings = torch.nn.MultiheadAttention(dim, head, batch_first=True)
        self.attn_root_children = torch.nn.MultiheadAttention(dim, head, batch_first=True)
        
        self.weight_leaf = nn.Parameter(torch.empty(3).normal_(mean=0,std=1))
        self.weight_inner = nn.Parameter(torch.empty(4).normal_(mean=0,std=1))
        self.weight_root = nn.Parameter(torch.empty(3).normal_(mean=0,std=1))
        
        self.softmax = torch.nn.Softmax(dim=-1)
        

    def forward(self, lir):
        leaf, inner, root = lir 

        leaf_p = self.attn_leaf_parents(leaf, inner.flatten(0,1).unsqueeze(1), inner.flatten(0,1).unsqueeze(1), need_weights=False)[0]
        leaf_s = self.attn_leaf_siblings(leaf, leaf, leaf, need_weights=False)[0]

        inner_p = self.attn_inner_parents(inner, root, root, need_weights=False)[0]
        inner_s = self.attn_inner_siblings(inner, inner, inner, need_weights=False)[0]
        inner_c = self.attn_inner_children(inner.flatten(0,1).unsqueeze(1), leaf, leaf, need_weights=False)[0].view(inner.shape)
        
        root_s = self.attn_root_siblings(root, root, root, need_weights=False)[0]
        root_c = self.attn_root_siblings(root, inner, inner, need_weights=False)[0]

        weight_leaf = self.softmax(self.weight_leaf)
        weight_inner = self.softmax(self.weight_inner)
        weight_root = self.softmax(self.weight_root)

        # NEED TO DO: Add Residual
        leaf = weight_leaf[0] * leaf_p + weight_leaf[1] * leaf_s + weight_leaf[2] * leaf
        inner = weight_inner[0] * inner_p + weight_inner[1] * inner_s + weight_inner[2] * inner_c + weight_inner[3] * inner
        root = weight_root[0] * root_s + weight_root[1] * root_c + weight_root[2] * root

        return [leaf, inner, root]

class AttnBlock(nn.Module):
    def __init__(self, dim, head, depth=3):
        super(AttnBlock, self).__init__()
        self.depth = depth
        self.dim = dim
        self.head = head

        self.mlp1s = []
        for i in range(self.depth):
            self.mlp1s.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, 1, bias=False),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.mlp1s = nn.ModuleList(self.mlp1s)

        self.attn2a = AttnLayer(self.dim, self.head)
        self.attn2bs = []
        for i in range(self.depth):
            self.attn2bs.append( 
                nn.Sequential(
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.attn2bs = nn.ModuleList(self.attn2bs)

        self.mlp3as = []
        for i in range(self.depth):
            self.mlp3as.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, 1, bias=False),
                    nn.BatchNorm1d(dim),
                )
            )
        self.mlp3as = nn.ModuleList(self.mlp3as)
        self.mlp3b = nn.ReLU(inplace=True)

    def forward(self, x_c):

        identity = []
        
        for x in x_c:
            identity.append(x)

        for i, x in enumerate(x_c):
            x_c[i] = self.mlp1s[i](x.permute(0,2,1)).permute(0,2,1)
        
        x_c = self.attn2a(x_c)

        for i, x in enumerate(x_c):
            x_c[i] = self.attn2bs[i](x.permute(0,2,1)).permute(0,2,1)

        for i, x in enumerate(x_c):
            x = self.mlp3as[i](x.permute(0,2,1)).permute(0,2,1)
            x = x + identity[i]
            x = self.mlp3b(x)

        return x_c

class HierAttnRes(nn.Module):
    def __init__(self, dim=128, head=4, num_blocks = 3):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = [AttnBlock(dim=dim, head=head) for _ in range(self.num_blocks)]
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, lir):
        leaf, inner, root = lir
        for b in self.blocks:
            leaf, inner, root = b([leaf, inner, root])
        return leaf, inner, root