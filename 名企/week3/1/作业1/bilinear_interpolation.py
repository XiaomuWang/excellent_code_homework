#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: bilinear_interplolation.py
@time: 2020/9/6 22:49
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def bilinear_interpolation(src_data, dst_size):
    """
    :param src_data: 输入图像矩阵
    :param dst_size: （dst_height, dst_width）
    :return: 输出图像
    """
    src_height, src_width, channel = src_data.shape
    dst_height, dst_width = dst_size
    ratio_height = float(src_height) / dst_height
    ratio_width = float(src_width) / dst_width
    dst_data = np.zeros((dst_height, dst_width, channel), np.uint8)
    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            # 目标在源上的坐标
            src_y = (dst_y + 0.5) * ratio_height - 0.5
            src_x = (dst_x + 0.5) * ratio_width - 0.5
            # 向下取整，计算在源图上四个近邻点的位置
            src_y_0 = int(src_y)
            src_x_0 = int(src_x)
            src_y_1 = min(src_y_0 + 1, src_height - 1)
            src_x_1 = min(src_x_0 + 1, src_width - 1)

            # 双线性插值
            #for i in range(channel):
            value_0 = (src_x_1 - src_x) * src_data[src_y_0, src_x_0] \
                      + (src_x - src_x_0) * src_data[src_y_0, src_x_1]
            value_1 = (src_x_1 - src_x) * src_data[src_y_1, src_x_0] \
                      + (src_x - src_x_0) * src_data[src_y_1, src_x_1]
            dst_data[dst_y, dst_x] = (src_y_1 - src_y) * value_0 + (src_y - src_y_0) * value_1
    return dst_data


if __name__ == "__main__":
    img = cv2.imread('./image.png')
    img2 = bilinear_interpolation(img, (800, 800))
    cv2.imwrite('./bilinear.png', img2)
    plt.imshow(img2)
    plt.show()
