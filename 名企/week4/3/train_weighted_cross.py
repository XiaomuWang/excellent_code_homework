#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_weighted_cross.py
@Time    :   2020/09/18 13:41:48
@Author  :   shengshijieshao
@Version :   1.0
@Contact :   2422656558@qq.com
'''


# 训练完成后，取最后一轮的权重保存，加权（中值频率平衡）交叉熵损失函数与原本的BCE损失函数（交叉熵损失函数的特例）相比提升幅度并不大。
# 但值得一提的是，通过对训练过程各指标的分析，可以看出BCEloss在后面几轮出现了过拟合的现象，而weighted_cross_entropy_loss没有出现明显过拟合
# 两个损失函数在最后的验证集上评价指标都差不多。甚至BCEloss更好一点。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.fcn import VGGNet, FCNs
from dataset.lane_cls_data import LaneClsDataset
from metrics.evaluator import Evaluator
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os


BATCH_SIZE = 4
LR = 1e-3
MAX_EPOCH = 50
IMG_H = 288
IMG_W = 800
SAVE_INTERVAL = 5
MODEL_CKPT_DIR = "./ckpt/median_frequency_balancing/"
FIGURE_DIR = "./figures/median_frequency_balancing/"


def draw_figure(record_dict, title="Loss", ylabel='loss', filename="loss.png"):
    plt.clf()
    epochs = np.arange(0, MAX_EPOCH)
    plt.plot(epochs, record_dict['train'], color='red', linewidth=1, label='train')
    plt.plot(epochs, record_dict['val'], color='blue', linewidth=1, label='val')
    plt.xticks(np.arange(0, MAX_EPOCH, 5))
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path)
    print('曲线图成功保存至{}'.format(save_path))


def main():
    # get model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fcn_model = FCNs(pretrained_net=VGGNet(pretrained=True, requires_grad=True))
    # criterion = nn.BCELoss()
    # 用带权重的交叉熵损失函数的话，需要先算出每个类的权重值
    # 这里使用SegNet里面提到的中值频率平衡（median frequency balancing, MFB）
    # 具体参考博客：https://blog.csdn.net/the_key0/article/details/89889243
    # https://blog.csdn.net/kinggang2017/article/details/92110844
    # 解释的最详细的在这：https://stats.stackexchange.com/questions/284265/understanding-median-frequency-balancing
    # 实现过程在median_frequency_balancing.py文件中
    weights = [3.26478624223462, 0.5904229036520159]
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights))
    optimizer = optim.Adam(fcn_model.parameters(), lr=LR)
    evaluator = Evaluator(num_class=2)

    if device == 'cuda':
        fcn_model.to(device)
        criterion.to(device)

    # get dataloader
    train_set = LaneClsDataset(list_path='train.tsv',
                               dir_path='data_road',
                               img_shape=(IMG_W, IMG_H))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8)

    val_set = LaneClsDataset(list_path='val.tsv',
                             dir_path='data_road',
                             img_shape=(IMG_W, IMG_H))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    # info records
    loss_dict = defaultdict(list)
    px_acc_dict = defaultdict(list)
    mean_px_acc_dict = defaultdict(list)
    mean_iou_dict = defaultdict(list)
    freq_iou_dict = defaultdict(list)

    for epoch_idx in range(1, MAX_EPOCH + 1):
        # train stage
        fcn_model.train()
        evaluator.reset()
        train_loss = 0.0
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            # print(label.shape)
            # label = label.reshape(BATCH_SIZE, 288, 800)
            label = label.to(device)
            optimizer.zero_grad()
            output = fcn_model(image)
            output = torch.sigmoid(output)

            loss = criterion(output, torch.argmax(label, dim=1).long())
            loss.backward()

            evaluator.add_batch(torch.argmax(output, dim=1).cpu().numpy(),
                                torch.argmax(label, dim=1).cpu().numpy())
            train_loss += loss.item()
            print("[Train][Epoch] {}/{}, [Batch] {}/{}, [Loss] {:.6f}".format(epoch_idx,
                                                                              MAX_EPOCH,
                                                                              batch_idx+1,
                                                                              len(train_loader),
                                                                              loss.item()))
            optimizer.step()
        loss_dict['train'].append(train_loss/len(train_loader))
        px_acc = evaluator.Pixel_Accuracy() * 100
        px_acc_dict['train'].append(px_acc)
        mean_px_acc = evaluator.Pixel_Accuracy_Class() * 100
        mean_px_acc_dict['train'].append(mean_px_acc)
        mean_iou = evaluator.Mean_Intersection_over_Union() * 100
        mean_iou_dict['train'].append(mean_iou)
        freq_iou = evaluator.Frequency_Weighted_Intersection_over_Union() * 100
        freq_iou_dict['train'].append(freq_iou)
        print("[Train][Epoch] {}/{}, [PA] {:.2f}%, [MeanPA] {:.2f}%, [MeanIOU] {:.2f}%, ""[FreqIOU] {:.2f}%".format(
            epoch_idx,
            MAX_EPOCH,
            px_acc,
            mean_px_acc,
            mean_iou,
            freq_iou))

        evaluator.reset()
        # validate stage
        fcn_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                output = fcn_model(image)
                output = torch.sigmoid(output)
                loss = criterion(output, torch.argmax(label, dim=1).long())
                val_loss += loss.item()
                evaluator.add_batch(torch.argmax(output, dim=1).cpu().numpy(),
                                    torch.argmax(label, dim=1).cpu().numpy())
            val_loss /= len(val_loader)
            loss_dict['val'].append(val_loss)
            px_acc = evaluator.Pixel_Accuracy() * 100
            px_acc_dict['val'].append(px_acc)
            mean_px_acc = evaluator.Pixel_Accuracy_Class() * 100
            mean_px_acc_dict['val'].append(mean_px_acc)
            mean_iou = evaluator.Mean_Intersection_over_Union() * 100
            mean_iou_dict['val'].append(mean_iou)
            freq_iou = evaluator.Frequency_Weighted_Intersection_over_Union() * 100
            freq_iou_dict['val'].append(freq_iou)
            print("[Val][Epoch] {}/{}, [Loss] {:.6f}, [PA] {:.2f}%, [MeanPA] {:.2f}%, "
                  "[MeanIOU] {:.2f}%, ""[FreqIOU] {:.2f}%".format(epoch_idx,
                                                                  MAX_EPOCH,
                                                                  val_loss,
                                                                  px_acc,
                                                                  mean_px_acc,
                                                                  mean_iou,
                                                                  freq_iou))

        # save model checkpoints
        if epoch_idx % SAVE_INTERVAL == 0 or epoch_idx == MAX_EPOCH:
            os.makedirs(MODEL_CKPT_DIR, exist_ok=True)
            ckpt_save_path = os.path.join(MODEL_CKPT_DIR, 'epoch_{}.pth'.format(epoch_idx))
            torch.save(fcn_model.state_dict(), ckpt_save_path)
            print("[Epoch] {}/{}, 模型权重保存至{}".format(epoch_idx, MAX_EPOCH, ckpt_save_path))

    # draw figures
    os.makedirs(FIGURE_DIR, exist_ok=True)
    draw_figure(loss_dict, title='Loss', ylabel='loss', filename='loss.png')
    draw_figure(px_acc_dict, title='Pixel Accuracy', ylabel='pa', filename='pixel_accuracy.png')
    draw_figure(mean_px_acc_dict, title='Mean Pixel Accuracy', ylabel='mean_pa', filename='mean_pixel_accuracy.png')
    draw_figure(mean_iou_dict, title='Mean IoU', ylabel='mean_iou', filename='mean_iou.png')
    draw_figure(freq_iou_dict, title='Freq Weighted IoU', ylabel='freq_weighted_iou', filename='freq_weighted_iou.png')


if __name__ == "__main__":
    main()
