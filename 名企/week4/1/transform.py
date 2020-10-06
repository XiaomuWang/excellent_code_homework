import cv2
import numpy as np
import random


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# 平移
def translation(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


class BGR2RGB(object):
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Resize(object):
    def __init__(self, height, width, interpolation):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        return cv2.resize(img, dsize=(self.width, self.height), interpolation=self.interpolation)


class HistogramNormalize(object):
    def __call__(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


class Mixup(object):
    def __init__(self, mask, random_translation, random_rotation):
        self.mask = mask
        self.random_translation = random_translation
        self.random_rotation = random_rotation

    def __call__(self, img):
        mask = cv2.resize(self.mask, (img.shape[1], img.shape[0]))
        if self.random_translation:
            x = random.random() * 100 - 50
            y = random.random() * 100 - 50
            mask = translation(mask, x, y)
        if self.random_rotation:
            angle = random.random() * 40 - 20
            mask = rotate(mask, angle)
        return (img * mask).astype(np.uint8)
