import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class KittiRoadTrainDataset(Dataset):
    BLUE_CHANNEL = 0
    GREEN_CHANEL = 1
    RED_CHANNEL = 2

    def __init__(self, path, type, img_transforms=[], gt_transforms=[]):
        super(KittiRoadTrainDataset, self).__init__()
        self.path = path
        self.img_transforms = img_transforms
        self.gt_transforms = gt_transforms
        # ground truth目录
        self.gt_path = os.path.join(path, 'gt_image_2')
        # 原始图片目录
        self.img_path = os.path.join(path, 'image_2')
        # 过滤道路类型和图片
        def filterType(f):
            return f.startswith(type + '_') and f.endswith('.png')
        # 拿到所有原始图片的名字
        self.images = list(filter(filterType, os.listdir(self.img_path)))

    def __getitem__(self, index):
        # 拿到图片个gt的路径
        img_path = os.path.join(self.img_path, self.images[index])
        gt_path = os.path.join(self.gt_path, self.images[index].replace('_', '_road_'))

        img = self.transform(cv2.imread(img_path), self.img_transforms)
        gt = self.transform(cv2.imread(gt_path), self.gt_transforms)
        gt_road = gt[self.BLUE_CHANNEL]
        gt_background = gt[self.RED_CHANNEL]

        # 返回图片、gt的数据，以及路径
        return (img, gt_road, gt_background, img_path, gt_path)

    def transform(self, img, transforms):
        for transform in transforms:
            img = transform(img)
        return img

    def __len__(self):
        return len(self.images)

class KittiRoadTestDataset(Dataset):
    def __init__(self, path, type, transforms=[]):
        super(KittiRoadTestDataset, self).__init__()
        self.path = path
        self.transforms = transforms
        # 原始图片目录
        self.img_path = os.path.join(path, 'image_2')
        # 过滤道路类型和图片
        def filterType(f):
            return f.startswith(type + '_') and f.endswith('.png')
        # 拿到所有原始图片的名字
        self.images = list(filter(filterType, os.listdir(self.img_path)))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.images[index])
        img = self.transform(cv2.imread(img_path), self.transforms)
        # 返回图片以及路径
        return (img, img_path)

    def transform(self, img, transforms):
        for transform in transforms:
            img = transform(img)
        return img

    def __len__(self):
        return len(self.images)