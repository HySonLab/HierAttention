import numpy as np
import random
import torch

def collate_fn(batch):
    coord, feat, label, cls, mapping = list(zip(*batch))
    offset, count = [], 0
    npoints, prev = [], 0
    ms = []
    for i, item in enumerate(coord):
        count += item.shape[0]
        offset.append(count)
        npoints.append(label[i].shape[0] + prev)
        m = mapping[i].clone()
        m[:, :2] += prev
        ms.append(m)
        prev = count
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), torch.IntTensor(cls), torch.cat(ms), torch.IntTensor(npoints)

