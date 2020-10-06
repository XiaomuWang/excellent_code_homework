#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/27 8:00

# @Author : ZFJ

# @File : train.py

# @Software: PyCharm
"""
import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TusimpleDataset
from lane_net_model import LaneNet
from utils.transforms import *
from utils.lr_scheduler import PolyLR

# 命令接口参数，方便命令行运行
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./exps")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

# resize_shape = (720, 1280)
# 因为本机现存不足，所以我只能用原来1/2的分辨率来进行训练
resize_shape = (360, 640)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------ train data ------------

BATCH_SIZE = 1
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Darkness(5), Rotation(2),
                          ToTensor(), Normalize(mean=mean, std=std))
train_dataset = TusimpleDataset("./data/", "train", transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=train_dataset.collate, num_workers=8)

# ------------ val data ------------
transform_val = Compose(Resize(resize_shape), ToTensor(),
                        Normalize(mean=mean, std=std))
val_dataset = TusimpleDataset("./data/", "val", transform_val)
# val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4)
# 现存不足，BS指定为1
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate, num_workers=4)

# ------------ preparation ------------
net = LaneNet(pretrained=True)
net = net.to(device)
net = torch.nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), lr=1e-3)
lr_scheduler = PolyLR(optimizer, 0.9, 30)
best_val_loss = 1e6


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_bin_seg = 0
    train_loss_var = 0
    train_loss_dist = 0

    progressbar = tqdm(range(len(train_loader)))

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        segLabel = sample['segLabel'].to(device)

        optimizer.zero_grad()
        output = net(img, segLabel)
        seg_loss = output['loss_seg']
        var_loss = output['loss_var']
        dist_loss = output['loss_dist']
        loss = output['loss']
        if isinstance(net, torch.nn.DataParallel):
            seg_loss = seg_loss.sum()
            var_loss = var_loss.sum()
            dist_loss = dist_loss.sum()
            loss = output['loss'].sum()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        train_loss_bin_seg += seg_loss.item()
        train_loss_var += var_loss.item()
        train_loss_dist += dist_loss.item()
        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)
        lr = optimizer.param_groups[0]['lr']

    progressbar.close()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))


def val(epoch):
    global best_val_loss

    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_bin_seg = 0
    val_loss_var = 0
    val_loss_dist = 0
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            segLabel = sample['segLabel'].to(device)

            output = net(img, segLabel)
            embedding = output['embedding']
            binary_seg = output['binary_seg']
            seg_loss = output['loss_seg']
            var_loss = output['loss_var']
            dist_loss = output['loss_dist']
            loss = output['loss']

            if isinstance(net, torch.nn.DataParallel):
                seg_loss = seg_loss.sum()
                var_loss = var_loss.sum()
                dist_loss = dist_loss.sum()
                loss = output['loss'].sum()

            val_loss += loss.item()
            val_loss_bin_seg += seg_loss.item()
            val_loss_var += var_loss.item()
            val_loss_dist += dist_loss.item()

            progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

    progressbar.close()
    print("------------------------\n")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth')
        copy_name = os.path.join(exp_dir, exp_dir.split('/')[-1] + '_best.pth')
        shutil.copyfile(save_name, copy_name)


def main():
    global best_val_loss
    if args.resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1e6)
    else:
        start_epoch = 0
    
    # for epoch in range(start_epoch, 100):
    # 饮食时间原因只跑了一个epoch作为模型测试，看看能否跑通
    for epoch in range(start_epoch, 1):
        train(epoch)
        if epoch % 2 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()
