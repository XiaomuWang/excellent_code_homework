#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@time: 2020/9/17 上午10:15
@file: split.py
@author: caoyuhua
@contact: caoyhseu@126.com
"""

import os
import shutil
import random
import re

train_image_dir = './training/image_2'
train_gt_dir = './training/gt_image_2'
valid_image_dir = './validing/image_2'
valid_gt_dir = './validing/gt_image_2'

if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
if not os.path.exists(valid_gt_dir):
    os.makedirs(valid_gt_dir)

image_list = os.listdir(train_image_dir)
valid_image_list = random.sample(image_list, int(0.1*len(image_list)))
for image_name in valid_image_list:
    image = os.path.join(train_image_dir, image_name)
    shutil.move(image, valid_image_dir)
    gt_image = os.path.join(train_gt_dir, re.sub('_', '_road_', image_name))
    shutil.move(gt_image, valid_gt_dir)
