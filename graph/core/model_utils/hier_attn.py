import torch
import torch.nn as nn

class AttnLayer(nn.Module):
    def __init__(self, dim, head):
        super(AttnLayer, self).__init__()
        self.attn_inner_siblings = torch.nn.MultiheadAttention(dim, head, batch_first=True, dropout=0.25)
        self.lin = nn.Linear(dim * 2, dim)
        #self.norm = nn.LayerNorm(dim)
        

    def forward(self, lir):
        leaf, inner, root = lir 

        inner_attn = self.attn_inner_siblings(inner, inner, inner, need_weights=False)[0]
        inner = torch.cat([inner_attn, inner], dim = -1)
        inner = self.lin(inner).relu()
        root = inner.sum(dim = 1)

        return [leaf, inner, root]

class AttnBlock(nn.Module):
    def __init__(self, dim, head, depth=3):
        super(AttnBlock, self).__init__()
        self.depth = depth
        self.dim = dim
        self.head = head

        self.attn2a = AttnLayer(self.dim, self.head)
        # self.attn2bs = []
        # for i in range(self.depth):
        #     self.attn2bs.append( 
        #         nn.Sequential(
        #             nn.BatchNorm1d(dim),
        #             nn.ReLU(inplace=True),
        #         )
        #     )
        # self.attn2bs = nn.ModuleList(self.attn2bs)

        # self.mlp3as = []
        # for i in range(self.depth):
        #     self.mlp3as.append(
        #         nn.Sequential(
        #             nn.Linear(dim, dim),
        #             nn.BatchNorm1d(dim),
        #             nn.ReLU(), 
        #             nn.Dropout(0.25),
        #             nn.Linear(dim, dim),
        #         )
        #     )
        # self.mlp3as = nn.ModuleList(self.mlp3as)
        # self.mlp3b = nn.ReLU(inplace=True)

        # self.mlp =  nn.Sequential(
        #             nn.Linear(dim, dim),
        #             nn.BatchNorm1d(dim),
        #             # nn.ReLU(), 
        #             # nn.Dropout(0.25),
        #             # nn.Linear(dim, dim),
        #         )

    def forward(self, x_c):

        # identity = []
        
        # for x in x_c:
        #     identity.append(x)
        
        x_c = self.attn2a(x_c)

        # for i, x in enumerate(x_c):
        #     x_c[i] = self.attn2bs[i](x.permute(0,2,1)).permute(0,2,1)

        #x_c[1] = x_c[1] + x_c[2] 

        # for i, x in enumerate(x_c[1:]):
        #     x = self.mlp3as[i+1][1](self.mlp3as[i+1][0](x).permute(0,2,1)).permute(0,2,1)
        #     x = x + identity[i+1]
        #     x = self.mlp3b(x)

        # x_c[-1] = self.mlp(x_c[-1])

        return x_c

class HierAttn(nn.Module):
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