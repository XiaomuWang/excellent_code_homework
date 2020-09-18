#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: nearest_neighbor_interplolation.py
@time: 2020/9/6 21:38
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def nearest_neighbor_interpolation(src_data, dst_size):
    """
    :param src_data: 输入图像矩阵
    :param dst_size: （dst_height, dst_width）
    :return: 输出图像
    """
    src_height, src_width, channel = src_data.shape
    dst_height, dst_width = dst_size
    ratio_height = src_height / dst_height
    ratio_width = src_width / dst_width
    dst_data = np.zeros((dst_height, dst_width, channel), np.uint8)
    #for i in range(channel):
    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            # 输出图片中坐标 （dst_x，dst_y）对应至输入图片中的（src_x，src_y）
            src_y = round(dst_y * ratio_height)  # 取整
            src_x = round(dst_x * ratio_width)
            # 防止四舍五入后越界
            if src_y >= src_height:
                src_y = src_y - 1
            if src_x >= src_width:
                src_x = src_x - 1
            # 插值
            dst_data[dst_y, dst_x] = src_data[src_y, src_x]
    return dst_data


if __name__ == "__main__":
    img = cv2.imread('./number_8.png')
    print(img.shape)
    img2 = nearest_neighbor_interpolation(img, (800, 800))
    cv2.imwrite('./nearest_neighbor.png', img2)
    plt.imshow(img2)
    plt.show()

