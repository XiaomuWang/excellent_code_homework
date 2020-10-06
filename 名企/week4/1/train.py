# lanhuajian

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from loss import BCEFocalLoss
from models import Fcn, Vgg11, ProxyNet
from dataset import KittiRoadTrainDataset
from transform import Resize
from torchvision.transforms import ToTensor
from torch.hub import load_state_dict_from_url
from evaluate import meanIou


def loadVgg11():
    vgg11 = Vgg11()
    # 获取预训练的参数
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth', progress=True)
    vgg11.load_state_dict(state_dict)

    return vgg11


def extract_shadow(img):
    # 随便提取了一些RGB的值，认为这些是阴影
    newImg = np.zeros_like(img)
    newImg[((30 < img[:, :, 0]) & (img[:, :, 0] < 40)) & ((30 < img[:, :, 1]) & (img[:, :, 1] < 60)) & (
                50 < img[:, :, 2]) & (img[:, :, 2] < 60)] = 1
    newImg[((40 < img[:, :, 0]) & (img[:, :, 0] < 50)) & ((40 < img[:, :, 1]) & (img[:, :, 1] < 50)) & (
                40 < img[:, :, 2]) & (img[:, :, 2] < 50)] = 1
    newImg[((30 < img[:, :, 0]) & (img[:, :, 0] < 40)) & ((30 < img[:, :, 1]) & (img[:, :, 1] < 40)) & (
                20 < img[:, :, 2]) & (img[:, :, 2] < 30)] = 1
    newImg[((20 < img[:, :, 0]) & (img[:, :, 0] < 30)) & ((20 < img[:, :, 1]) & (img[:, :, 1] < 30)) & (
                20 < img[:, :, 2]) & (img[:, :, 2] < 30)] = 1
    return newImg


if __name__ == '__main__':
    ITERATION, BATCH_SIZE, LR, CLASSES_NUM, = 100, 10, 1e-5, 2
    # 缩小图片尺寸，让计算速度更快一些，要注意尺寸要是32（vgg会下采样5次）的整数倍
    HEIGHT, WIDTH = 288, 800
    USE_FOCAL_LOSS = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = None
    if USE_FOCAL_LOSS:
        # focal loss中我传入的是道路的概率值，道路的像素数与背景的像素数之比大约为1:4
        # 所以应该取道路权重取0.80，背景权重取0.20， 这样就能提高道路部分的权重，降低背景的权重，
        # 求loss时会让道路部分的梯度更大，背景梯度更小
        criterion = BCEFocalLoss(alpha=0.20)
        criterion.to(device)
        # 直接用一个channel来代表每个像素是否是道路的概率，减少计算量
        # CLASSES_NUM = 1
    else:
        # 交叉熵权重，要与输入图像大小一致
        # 可以发现最开始时，不用weight的loss比用weight的要大
        lossWeight = torch.ones((CLASSES_NUM, HEIGHT, WIDTH))
        # 第一个channel是识别道路的，面积比较小，但更加重要，所以权重要大一些
        lossWeight[0] = 0.8
        # 第二个channel是识别背景的，面积比较大，不那么重要，所以权重要小一些
        lossWeight[1] = 0.2
        # BCELoss是专门用于做二分类的交叉熵，如果用于多分类则需要CrossEntropyLoss
        criterion = nn.BCELoss(weight=lossWeight)
    criterion.to(device)

    # 构造vgg11，并且加载torch提供的预训练参数
    vgg11 = loadVgg11()
    # 只需要cgg11.features（含下采样）来构造一个代理的网络，并记录每一层的结果
    # 不需要vgg11.classifier
    fineTuringNet = ProxyNet('vgg11', vgg11.features)
    # 根据vgg11的features来构造fcn8，输出道路和背景两个channel
    model = Fcn(scale=8, featureProxyNet=fineTuringNet, classesNum=CLASSES_NUM)
    model = model.to(device)

    # 无脑选择Adam优化
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 随便找一张图来提取阴影部分，用来叠加到其他图片来做数据增强
    # maskImg = cv2.imread('./data_road/testing/image_2/um_000087.png')
    # maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2RGB)
    # shadowImg = extract_shadow(maskImg)
    # shadowImg = shadowImg.astype(np.float32)
    # 阴影部分，让原图的像素值变为原来的1/5，其他部分保持不变
    # shadowImg[shadowImg == 1] = 0.2
    # shadowImg[shadowImg == 0] = 1

    # 加载训练数据集
    trainSet = KittiRoadTrainDataset(path='./data_road/training', type='um', img_transforms=[
        # 缩小图片节省计算量，并且保证能被32整除
        Resize(HEIGHT, WIDTH, cv2.INTER_LINEAR),
        # 直方图均衡
        # HistogramNormalize(),
        # 叠加一些阴影噪声，对阴影进行随机平移和旋转
        # Mixup(shadowImg, random_translation=True, random_rotation=True),
        # ndarray转张量
        ToTensor(),
    ], gt_transforms=[
        Resize(HEIGHT, WIDTH, cv2.INTER_LINEAR),
        ToTensor(),
    ])
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    BATCH = len(trainLoader)
    for iter in range(ITERATION):
        mean_ious = []
        for batch, (img, gt_road, gt_background, img_path, gt_path) in enumerate(trainLoader):
            # 以下是训练代码的通用套路
            # 梯度清零
            optimizer.zero_grad()
            img = img.to(device)
            # 前向推理
            output = model(img)
            loss = mIou = None
            if CLASSES_NUM == 1:
                # 当分类数是1的时候，认为输出只有一个channel，每个位置上的像素值代表是否是道路的概率
                gt_road = gt_road.to(device)
                loss = criterion(output, gt_road.reshape((gt_road.shape[0], 1, HEIGHT, WIDTH)))
            elif CLASSES_NUM == 2:
                # 对输出结果取sigmoid，把道路和背景两个channel的数值归一化到[0,1]之间，用来代表属于该类的概率
                output = torch.sigmoid(output)
                # 组合道路和背景两个channel，新增维度到第二个位置
                # 变为[batch, channel, height, width]
                gt = torch.stack((gt_road, gt_background), 1)
                gt = gt.to(device)
                # 用交叉熵来算loss，这样求参数的梯度时能够把sigmoid的部分消掉，避免sigmoid两头梯度消失
                loss = criterion(output, gt)
                mIou = meanIou(output, gt)
            # 求参数梯度
            loss.backward()
            print("iter: {}/{}, batch: {}/{}, meanIou: {:.5f}, loss: {:.5f}".format(iter + 1, ITERATION, batch + 1, BATCH, mIou, loss.item()))
            # 更新参数
            optimizer.step()

        if iter % 10 == 9:
            print("save model")
            torch.save(model.state_dict(), 'fcn_road_segment_{}.pth'.format(iter))

    torch.save(model.state_dict(), 'fcn_road_segment.pth'.format(iter))
    print("ok")