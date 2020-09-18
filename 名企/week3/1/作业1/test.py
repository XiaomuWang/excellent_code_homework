#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: test.py
@time: 2020/9/7 17:51
'''

import cv2
from nearest_neighbor_interpolation import nearest_neighbor_interpolation
from bicubic_interpolation import bicubic_interpolation
from bilinear_interpolation import bilinear_interpolation
import time

if __name__ == "__main__":
    save_path = './result'
    img = cv2.imread('./image.png')

    t1 = time.time()
    img11 = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
    t2 = time.time()
    print("使用opencv中最近邻插值需要的时间为：{:.6f}s".format(t2-t1))
    cv2.imwrite('{}/nearest_neighbor_opencv.png'.format(save_path), img11)

    t1 = time.time()
    img12 = nearest_neighbor_interpolation(img, (800, 800))
    t2 = time.time()
    print("使用自己编写的最近邻插值需要的时间为：{:.6f}s".format(t2 - t1))
    cv2.imwrite('{}/nearest_neighbor_own.png'.format(save_path), img12)

    t1 = time.time()
    img21 = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
    t2 = time.time()
    print("使用opencv中双线性插值需要的时间为：{:.6f}s".format(t2 - t1))
    cv2.imwrite('{}/bilinear_opencv.png'.format(save_path), img21)

    t1 = time.time()
    img22 = bilinear_interpolation(img, (800, 800))
    t2 = time.time()
    print("使用自己编写的双线性插值需要的时间为：{:.6f}s".format(t2 - t1))
    cv2.imwrite('{}/bilinear_own.png'.format(save_path), img22)

    t1 = time.time()
    img31 = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
    t2 = time.time()
    print("使用opencv中双三次插值需要的时间为：{:.6f}s".format(t2 - t1))
    cv2.imwrite('{}/bicubic_opencv.png'.format(save_path), img31)

    t1 = time.time()
    img32 = bicubic_interpolation(img, (800, 800))
    t2 = time.time()
    print("使用自己编写的双三次插值需要的时间为：{:.6f}s".format(t2 - t1))
    cv2.imwrite('{}/bicubic_own.png'.format(save_path), img32)
