from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from util.ModelNet40Tree import ModelNetDataLoaderTree
# from util.intra3d import Intra3D
import numpy as np
from torch.utils.data import DataLoader, Subset
from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list
import util.provider as provider
import sklearn.metrics as metrics
import random
import wandb
from tqdm import tqdm
import time


def get_dataloader(args, split):
    name = args.data_name
    if name  == 'modelnet40':
        return ModelNetDataLoaderTree(path=args.data_root, npoints=args.num_point, split=split, use_normals=args.pt_norm, process_data=args.process_data, depth = args.depth, divide = args.divide)
    elif name == 'intra3d': 
        return Intra3D(path=args.data_root, split=split, npoints=args.num_point, cls_state=True, rotate = rotate, shift = shift)
    else: 
        return None 

def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='', help='config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_weights.t7', help='Checkpoint')
    parser.add_argument('--test', action="store_true", help='Test')
    parser.add_argument('--subset', action="store_true", help='Use only subset of data to test')
    parser.add_argument('--wandb', action="store_true", help='Report to wandb')
    parser.add_argument('--process_data', action="store_true", help='Process and save dataloader so that the later run will load data faster')

    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 8)
    cfg['subset'] = args.subset
    cfg['wandb'] = args.wandb
    cfg['process_data'] = args.process_data
    cfg['checkpoint'] = args.checkpoint
    cfg['test'] = args.test

    return cfg


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)

    if not args.eval:  # backup the running files
        if args.arch == 'point-transformer':
             os.system('cp model/PointTransformer.py checkpoints' + '/' + args.exp_name + '/' + 'PointTransfomer.py.backup')

# weight initialization:
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def collate_fn(batch):
    coord, feat, mapping, label = list(zip(*batch))
    offset, count = [], 0
    ms = []
    prev = 0
    for i, item in enumerate(coord):
        count += item.shape[0]
        offset.append(count)
        m = mapping[i].clone()
        m[:, :2] += prev
        ms.append(m)
        prev = count

    return torch.cat(coord), torch.cat(feat), torch.cat(ms), torch.cat(label), torch.IntTensor(offset)
    

def train(args, io):

    # Get subset of data to test logging
    if args.subset:
        print("Use subset of data")
        train_loader = DataLoader(Subset(get_dataloader(args, "training"), list(range(16))),
                                num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        test_loader = DataLoader(Subset(get_dataloader(args, "validation"),list(range(16))),
                                num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        # test_loader_equivariant = DataLoader(Subset(get_dataloader(args, "validation_equivariant"),list(range(16))),
        #                         num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(get_dataloader(args, "training"),
                                num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        test_loader = DataLoader(get_dataloader(args, "validation"),
                                num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        # test_loader_equivariant = DataLoader(get_dataloader(args, "validation_equivariant"),
        #                         num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.arch == 'tree':
        from model.TreeTransformerCls import TreeTransformerCls
        model = TreeTransformerCls(args).to(device)
    else:
        raise Exception("Not implemented")

    # io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    print(args.workers)
    
    print("Use SGD")
    opt = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001
        )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

    criterion = cal_loss

    best_test_acc = 0
    if(args.wandb):
        wandb.watch(model)
        
    for epoch in range(args.epochs):
        io.cprint("Epoch: " + str(epoch + 1))
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        for i, (coord, feat, mapping, label, offset) in enumerate(tqdm(train_loader)):
            coord, feat, mapping, label, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), mapping.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze().long(), offset.cuda(non_blocking=True) 
            opt.zero_grad()
            logits = model([coord, feat, offset, mapping])
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += args.batch_size
            train_loss += loss.item() * args.batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, ' % (epoch + 1, train_loss * 1.0 / count, train_acc)
        io.cprint(outstr)

        if(args.wandb):
            wandb.log({
                        "Loss train": train_loss * 1.0 / count,
                        "Accuracy train": train_acc
                    },
                    step = epoch + 1)


        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []

        for i, (coord, feat, mapping, label, offset) in enumerate(tqdm(test_loader)):
            coord, feat, mapping, label, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), mapping.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze().long(), offset.cuda(non_blocking=True) 
            logits = model([coord, feat, offset, mapping])

            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += args.batch_size
            test_loss += loss.item() * args.batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)

        outstr = 'Test %d, loss: %.6f, test acc: %.6f,' % (epoch + 1, test_loss * 1.0 / count, test_acc)
        io.cprint(outstr)

        if(args.wandb):
            wandb.log({
                "Loss test": test_loss * 1.0 / count,
                "Accuracy test": test_acc
            },
            step = epoch + 1)

        print(test_acc)
        print(best_test_acc)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            io.cprint('Max Acc:%.6f' % best_test_acc)
            torch.save(model.state_dict(), 'checkpoints/best_model_weights.t7')
        
def test(args, io):
    num_class = 40
    vote_num = 1
    test_loader = DataLoader(get_dataloader(args, "validation"),
                                num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models:
    if args.arch == 'tree':
        from model.TreeTransformerCls import TreeTransformerCls
        model = TreeTransformerCls(args).to(device)
    else:
        raise Exception("Not implemented")

    # io.cprint(str(model))

    model = nn.DataParallel(model)
    if not args.checkpoint is None:
        model.load_state_dict(torch.load(args.checkpoint))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    times = []
    mean_correct = []
    model = model.eval()
    class_acc = np.zeros((num_class, 3))

    for i, (coord, feat, mapping, label, offset) in enumerate(tqdm(test_loader)):
        coord, feat, mapping, label, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), mapping.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze().long(), offset.cuda(non_blocking=True)
        vote_pool = torch.zeros(label.size()[0], num_class).cuda()

        for _ in range(vote_num):
            with torch.no_grad():
                pred = model([coord, feat, offset, mapping])
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(label.cpu()):
            classacc = pred_choice[label == cat].eq(label[label == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(label[label == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(label.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(label.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    print(instance_acc, class_acc)


if __name__ == "__main__":
    args = get_parser()
    _init_()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if(args.wandb):
        wandb.init(project=args.wandb_project, entity="mila-pointcloud", name=args.exp_name, config=args)
    
    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.test:
        train(args, io)
    else:
        test(args, io)
