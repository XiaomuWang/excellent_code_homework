#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: bicubic_interplolation.py
@time: 2020/9/7 18:10
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 插值内核
def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0


# 补边，将输入图像矩阵前后各扩展两行两列,当插值点为边沿点时候，确保周围有16个点
def padding(image, H, W, C):
    """
    :param image: 输入图像
    :param H: 高
    :param W: 宽
    :param C: 通道数
    :return: 补边后的图像
    """
    padding_image = np.zeros((H + 4, W + 4, C))
    padding_image[2: H + 2, 2: W + 2, :] = image

    # 分别对图像前后各两行、左右各两列进行补边
    padding_image[2: H + 2, 0: 2, :] = image[:, 0: 1, :]  # 左侧两列
    padding_image[2: H + 2, -2:, :] = image[:, -1:, :]  # 右侧两列
    padding_image[0: 2, 2: W + 2, :] = image[0, :, :]  # 上边两行
    padding_image[-2:, 2: W + 2, :] = image[-1, :, :]  # 下边两行

    # 对四个角上的16个点进行填充
    padding_image[0: 2, 0: 2, :] = image[0, 0, :]      # 左上侧
    padding_image[-2: 0, 0: 2, :] = image[-1, 0, :]    # 左下侧
    padding_image[-2: 0, -2: 0, :] = image[-1, -1, :]  # 右下侧
    padding_image[0: 2, -2: 0, :] = image[0, -1, :]    # 右上侧

    return padding_image


def bicubic_interpolation(src_data, dst_size, a=-0.5):
    """
    :param a:
    :param src_data: 输入图像矩阵
    :param dst_size: （dst_height, dst_width）
    :return: 输出图像
    """
    src_height, src_width, channel = src_data.shape
    dst_height, dst_width = dst_size

    src_data = padding(src_data, src_height, src_width, channel)  # 补边操作

    ratio_height = float(src_height) / dst_height
    ratio_width = float(src_width) / dst_width
    dst_data = np.zeros((dst_height, dst_width, channel), np.uint8)

    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            # 目标在源上的坐标
            src_y = dst_y * ratio_height + 2  # 加2是因为上面扩大了四行四列，要回到原来图像的点再计算
            src_x = dst_x * ratio_width + 2
            # 16个点距源点的距离
            x1 = 1 + src_x - int(src_x)
            x2 = src_x - int(src_x)
            x3 = int(src_x) + 1 - src_x
            x4 = int(src_x) + 2 - src_x

            y1 = 1 + src_y - int(src_y)
            y2 = src_y - int(src_y)
            y3 = int(src_y) + 1 - src_y
            y4 = int(src_y) + 2 - src_y

            mat_l = np.array([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])  # 四个横坐标的权重
            mat_r = np.array([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])  # 四个纵坐标的权重
            for i in range(channel):
                mat_m = np.array([[src_data[int(src_y - y1), int(src_x - x1), i],
                                   src_data[int(src_y - y2), int(src_x - x1), i],
                                   src_data[int(src_y + y3), int(src_x - x1), i],
                                   src_data[int(src_y + y4), int(src_x - x1), i]],

                                   [src_data[int(src_y - y1), int(src_x - x2), i],
                                    src_data[int(src_y - y2), int(src_x - x2), i],
                                    src_data[int(src_y + y3), int(src_x - x2), i],
                                    src_data[int(src_y + y4), int(src_x - x2), i]],

                                   [src_data[int(src_y - y1), int(src_x + x3), i],
                                    src_data[int(src_y - y2), int(src_x + x3), i],
                                    src_data[int(src_y + y3), int(src_x + x3), i],
                                    src_data[int(src_y + y4), int(src_x + x3), i]],

                                   [src_data[int(src_y - y1), int(src_x + x4), i],
                                    src_data[int(src_y - y2), int(src_x + x4), i],
                                    src_data[int(src_y + y3), int(src_x + x4), i],
                                    src_data[int(src_y + y4), int(src_x + x4), i]]])

                dst_data[dst_y, dst_x] = np.clip(np.dot(np.dot(mat_l, mat_m), mat_r), 0, 255)
    return dst_data


if __name__ == "__main__":
    img = cv2.imread('./image.png')
    img3 = bicubic_interpolation(img, (800, 800))
    cv2.imwrite('./bicubic.png', img3)
    plt.imshow(img3)
    plt.show()
