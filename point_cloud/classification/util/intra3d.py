import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
import util.provider as provider

class Intra3D(Dataset):
    def __init__(self, path, split='training', cls_state=False, npoints=2048, choice=1, shift=False, rotate=False):
        self.shift = shift
        self.rotate = rotate
        
        self.npoints = npoints  # 2048 pts
        self.datapath = []
        self.label = {}
        self.cls_state = cls_state
        self.train_mode = split
        BASE = path
        choice_list = [i for i in range(5)]
        poped_val = choice_list.pop(choice)

        if self.cls_state:
            self.label[0] = glob.glob(BASE + "generated/vessel/ad/" + "*.ad")  # label 0: healthy; 1694 files; negSplit
            self.label[1] = glob.glob(BASE + "generated/aneurysm/ad/" + "*.ad") + \
                            glob.glob(BASE + "annotated/ad/" + "*.ad")  # label 1: unhealthy; 331 files

            train_test_set_ann = natsorted(glob.glob(BASE + "fileSplit/cls/ann_clsSplit_" + "*.txt"))  # label 1
            train_test_set_neg = natsorted(glob.glob(BASE + "fileSplit/cls/negSplit_" + "*.txt"))  # label 0
            train_set = [train_test_set_ann[i] for i in choice_list] + [train_test_set_neg[i] for i in choice_list]
            test_set = [train_test_set_ann[poped_val]] + [train_test_set_neg[poped_val]]
        else:
            train_test_set = natsorted(glob.glob(BASE + "fileSplit/seg/annSplit_" + "*.txt"))
            train_set = [train_test_set[i] for i in choice_list]
            test_set = [train_test_set[poped_val]]

        if self.train_mode == 'training':
            for file in train_set:
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        elif self.train_mode == 'validation':
            for file in test_set:
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        elif self.train_mode == 'all':
            for file in (train_set + test_set):
                with open(file, 'r') as f:
                    for line in f.readlines():
                        self.datapath.append(BASE + line.strip())
        else:
            print("Error")
            raise Exception("training mode invalid")

    def __getitem__(self, index):
        curr_file = self.datapath[index]
        cls = None
        if self.cls_state:
            if curr_file in self.label[0]:
                cls = torch.from_numpy(np.array([0]).astype(np.int64))
            elif curr_file in self.label[1]:
                cls = torch.from_numpy(np.array([1]).astype(np.int64))
            else:
                print("Error found!!!")
                exit(-1)

        point_set = np.loadtxt(curr_file)[:, :-1].astype(np.float32)  # [x, y, z, norm_x, norm_y, norm_z]
        seg = np.loadtxt(curr_file)[:, -1].astype(np.int64)  # [seg_label]
        seg[np.where(seg == 2)] = 1  # making boundary lines (label 2) to A. (label 1)

        # random choice
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # normalization to unit ball
        point_set[:, :3] = point_set[:, :3] - np.mean(point_set[:, :3], axis=0)  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)
        point_set[:, :3] = point_set[:, :3] / dist

        if self.rotate:
            point_set[:, :3] = provider.rotate_point_cloud_random(point_set[:, :3])

        if self.shift:
            point_set[:, :3] = provider.shift_point_cloud(point_set[:, :3])

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)

        if self.cls_state:
            return point_set[..., :3], point_set[..., 3:6], cls
        else:
            return point_set[..., :3], point_set[..., 3:6], seg

    def __len__(self):
        return len(self.datapath)

