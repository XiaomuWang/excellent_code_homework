# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/9/27

import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from project_week5.lanenet.model.loss import discriminative_loss
from project_week5.lanenet.dataset import TuSimpleDataset


def init_args():
    """
    参数设定
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/train_set', help='path to TuSimple Benchmark dataset')
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type(default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use seperate decoders for two branches')
    parser.add_argument('--tag', type=str, help='training tag')
    return parser.parse_args()


def init_weights(model):
    """
    初始化模型权重
    :param model: 输入模型
    """
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


if __name__ == '__main__':

    # 忽略 warning 输出
    warnings.filterwarnings(action="ignore", module='torch')

    # 加载参数
    args = init_args()

    # 数据集均值
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)
    VGG_MEAN = torch.from_numpy(VGG_MEAN).cuda().view([1, 3, 1, 1])

    # 超参数设置
    batch_size = 64  # batch size per GPU
    learning_rate = 1e-3  # 1e-3
    num_steps = 100
    num_workers = 4
    # 设置整个训练过程中每完整训练50次保存一次模型
    ckpt_epoch_interval = 50
    # 设置每迭代50次验证一次模型
    val_step_interval = 50
    # 记录开始训练时间
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    # 判断是否使用GPU
    device = torch.device("cuda:0")

    # 加载数据集
    data_dir = args.data_dir
    train_set = TuSimpleDataset(data_dir, 'train')
    val_set = TuSimpleDataset(data_dir, 'val')

    num_train = len(train_set)
    num_val = len(val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders = {'train': train_loader, 'val': val_loader}
    print('Finish loading data from %s' % data_dir)

    # 利用tensorboard可视化训练过程，并设置日志文件保存位置
    writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    # 选择网络结构
    arch = args.arch
    if 'fcn' in arch.lower():
        arch = 'lanenet.LaneNet_FCN_Res'
    elif 'enet' in arch.lower():
        arch = 'lanenet.LaneNet_ENet'
    elif 'icnet' in arch.lower():
        arch = 'lanenet.LaneNet_ICNet'

    # 判断使用一个编码器还是两个编码器（编码器权值是否共享）
    arch = arch + '_1E2D' if args.dual_decoder else arch + '_1E1D'
    print('Architecture:', arch)
    # 自动对应模型名称到模型
    net = eval(arch)()

    # 若有多张GPU则并行模型
    # net = nn.DataParallel(net)

    # 将模型移到GPU上
    net.to(device)

    # 设定优化器及损失函数相关
    params_to_update = net.parameters()
    optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=0.0001)
    MSELoss = nn.MSELoss()

    # 是否加载与训练模型
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # step = checkpoint['step']
        step = 0  # by default, we reset step and epoch value
        epoch = 1
        loss = checkpoint['loss']
        print('Checkpoint loaded.')

    else:
        # 若无预训练模型则初始化模型参数
        net.apply(init_weights)
        step = 0
        epoch = 1
        print('Network parameters initialized.')
    
    # 初始化训练过程指标数据
    sum_bin_precision_train, sum_bin_precision_val = 0, 0
    sum_bin_recall_train, sum_bin_recall_val = 0, 0
    sum_bin_F1_train, sum_bin_F1_val = 0, 0

    # 将数据封装为迭代器
    data_iter = {'train': iter(dataloaders['train']), 'val': iter(dataloaders['val'])}
    for step in range(step, num_steps):
        # 记录训练开始时间
        start_time = time.time()

        phase = 'train'
        net.train()
        if step % val_step_interval == 0:
            phase = 'val'
            net.eval()

        # 加载一个batch的数据
        try:
            batch = next(data_iter[phase])
        except StopIteration:
            # 迭代完一次数据集
            data_iter[phase] = iter(dataloaders[phase])
            batch = next(data_iter[phase])

            if phase == 'train':
                epoch += 1

                # 保存模型
                if epoch % ckpt_epoch_interval == 0:
                    # 设置模型保存路径
                    ckpt_dir = 'check_point/ckpt_%s_%s' % (train_start_time, args.tag)
                    if os.path.exists(ckpt_dir) is False:
                        os.mkdir(ckpt_dir)
                    ckpt_path = os.path.join(ckpt_dir, 'ckpt_%s_epoch-%d.pth' % (train_start_time, epoch))
                    # 保存模型及此时训练阶段相关状态
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, ckpt_path)

                # 计算训练过程相关指标并将其添加到日志中
                avg_precision_bin_train = sum_bin_precision_train / num_train
                avg_recall_bin_train = sum_bin_recall_train / num_train
                avg_F1_bin_train = sum_bin_F1_train / num_train
                writer.add_scalar('Epoch_Precision_Bin-TRAIN', avg_precision_bin_train, step)
                writer.add_scalar('Epoch_Recall_Bin-TRAIN', avg_recall_bin_train, step)
                writer.add_scalar('Epoch_F1_Bin-TRAIN', avg_F1_bin_train, step)
                writer.add_text('Epoch', str(epoch), step)
                sum_bin_precision_train = 0
                sum_bin_recall_train = 0
                sum_bin_F1_train = 0

            elif phase == 'val':
                # 计算验证过程相关指标并将其添加到日志中
                avg_precision_bin_val = sum_bin_precision_val / num_val
                avg_recall_bin_val = sum_bin_recall_val / num_val
                avg_F1_bin_val = sum_bin_F1_val / num_val
                writer.add_scalar('Epoch_Precision_Bin-VAL', avg_precision_bin_val, step)
                writer.add_scalar('Epoch_Recall_Bin-VAL', avg_recall_bin_val, step)
                writer.add_scalar('Epoch_F1_Bin-VAL', avg_F1_bin_val, step)
                sum_bin_precision_val = 0
                sum_bin_recall_val = 0
                sum_bin_F1_val = 0

        # 解析出具体数据及标签
        inputs = batch['input_tensor']
        labels_bin = batch['binary_tensor']
        labels_inst = batch['instance_tensor']
        # 将数据迁移到GPU上，有的话
        inputs = inputs.to(device)
        labels_bin = labels_bin.to(device)
        labels_inst = labels_inst.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        embeddings, logit = net(inputs)

        # 计算损失
        preds_bin = torch.argmax(logit, dim=1, keepdim=True)
        preds_bin_expand = preds_bin.view(preds_bin.shape[0] * preds_bin.shape[1] * preds_bin.shape[2] * preds_bin.shape[3])
        labels_bin_expand = labels_bin.view(labels_bin.shape[0] * labels_bin.shape[1] * labels_bin.shape[2])

        # Floating Loss weighting determined by label proportion
        bin_count = torch.bincount(labels_bin_expand)
        bin_prop = bin_count.float() / torch.sum(bin_count)
        weight_bin = torch.tensor(1) / (bin_prop + 0.2)  # max proportion: 5:1

        # 计算语义分割损失
        # Multi-class CE Loss
        CrossEntropyLoss = nn.CrossEntropyLoss(weight=weight_bin)
        loss_bin = CrossEntropyLoss(logit, labels_bin)

        # discriminative loss
        loss_disc, loss_v, loss_d, loss_r = discriminative_loss(embeddings,
                                                                labels_inst,
                                                                delta_v=0.2,
                                                                delta_d=1,
                                                                param_var=.5,
                                                                param_dist=.5,
                                                                param_reg=0.001)

        # 损失函数总和，0.01参数可调
        loss = loss_bin + loss_disc * 0.01

        # 反向传播并更新梯度
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # 计算评价指标， TP， precision, recall, F1
        bin_TP = torch.sum((preds_bin_expand.detach() == labels_bin_expand.detach()) & (preds_bin_expand.detach() == 1))
        bin_precision = bin_TP.double() / (torch.sum(preds_bin_expand.detach() == 1).double() + 1e-6)
        bin_recall = bin_TP.double() / (torch.sum(labels_bin_expand.detach() == 1).double() + 1e-6)
        bin_F1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall)

        # 计算一次迭代所耗费时间
        step_time = time.time() - start_time

        if phase == 'train':
            step += 1

            # 累加相关结果
            sum_bin_precision_train += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_train += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_train += bin_F1.detach() * preds_bin.shape[0]

            # 输出相关loss和指标到日志文件
            writer.add_scalar('learning_rate', learning_rate, step)
            writer.add_scalar('total_train_loss', loss.item(), step)
            writer.add_scalar('bin_train_loss', loss_bin.item(), step)
            writer.add_scalar('bin_train_F1', bin_F1, step)
            writer.add_scalar('disc_train_loss', loss_disc.item(), step)

            # 打印出训练过程中相关指标
            print('{}  {}  \nEpoch:{}  Step:{}  TrainLoss:{:.5f}  Bin_Loss:{:.5f}  '
                  'BinRecall:{:.5f}  BinPrec:{:.5f}  F1:{:.5f}  '
                  'DiscLoss:{:.5f}  vLoss:{:.5f}  dLoss:{:.5f}  rLoss:{:.5f}  '
                  'Time:{:.2f}'
                  .format(train_start_time, args.tag, epoch, step, loss.item(), loss_bin.item(),
                          bin_recall.item(), bin_precision.item(), bin_F1.item(),
                          loss_disc.item(), loss_v.item(), loss_d.item(), loss_r.item(),
                          step_time))

        elif phase == 'val':
            sum_bin_precision_val += bin_precision.detach() * preds_bin.shape[0]
            sum_bin_recall_val += bin_recall.detach() * preds_bin.shape[0]
            sum_bin_F1_val += bin_F1.detach() * preds_bin.shape[0]

            writer.add_scalar('total_val_loss', loss.item(), step)
            writer.add_scalar('bin_val_loss', loss_bin.item(), step)
            writer.add_scalar('bin_val_F1', bin_F1, step)
            writer.add_scalar('disc_val_loss', loss_disc.item(), step)

            print('\n{}  {}  \nEpoch:{}  Step:{}  ValidLoss:{:.5f}  BinLoss:{:.5f}  '
                  'BinRecall:{:.5f}  BinPrec:{:.5f}  F1:{:.5f}  '
                  'DiscLoss:{:.5f}  vLoss:{:.5f}  dLoss:{:.5f}  rLoss:{:.5f}  '
                  'Time:{:.2f}'
                  .format(train_start_time, args.tag, epoch, step, loss.item(), loss_bin.item(),
                          bin_recall.item(), bin_precision.item(), bin_F1.item(),
                          loss_disc.item(), loss_v.item(), loss_d.item(), loss_r.item(),
                          step_time))

            # 保存验证结果
            num_images = 3  # Select the number of images to be saved in each val iteration
            # 逆变换数据并交换通道
            inputs_images = (inputs + VGG_MEAN / 255.)[:num_images, [2, 1, 0], :, :]

            writer.add_images('image', inputs_images, step)
            writer.add_images('Bin Pred', preds_bin[:num_images], step)

            labels_bin_img = labels_bin.view(labels_bin.shape[0], 1, labels_bin.shape[1], labels_bin.shape[2])
            writer.add_images('Bin Label', labels_bin_img[:num_images], step)

            embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5  # a tricky way to visualize the embedding
            writer.add_images('Embedding', embedding_img, step)
