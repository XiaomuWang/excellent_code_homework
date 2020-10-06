#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   median_frequency_balancing.py
@Time    :   2020/09/17 20:30:43
@Author  :   shengshijieshao
@Version :   1.0
@Contact :   2422656558@qq.com
'''


'''

classPixelCount = [array of class.size() zeros]
classTotalCount = [array of class.size() zeros]

for each image in dataset:
    perImageFrequencies = bincount(image)
    classPixelCount = element_wise_sum(classPixelCount, perImageFrequencies)
    nPixelsInImage = image.total_pixel_count()
    for each frequency in per_image_frequencies:
        if frequency > 0:
            classTotalCount = classTotalCount + nPixelsInImage

return elementwiseDivision(classPixelCount, classTotalCount)

Finally, to calculate the class weights: 
sortedFrequencies = sort(frequences) 
medianFreq = median(frequencies) 
return elementwiseDivision(medianFreq, sortedFrequencies)
'''

import numpy as np
from PIL import Image
import glob

# 将原始标注里的红色，紫色，黑色三色标签转换成黑白二色，其中表述车道的紫色为白色，其他部分为黑色。
# 其实这个地方不太严谨哈，因为黑色好像表示的是车道线。在我们训练的过程中，也是将红色看作背景类，其他颜色看作车道线这一类的。
def change_label(image):
    w, h = image.size
    for x in range(w):
        for y in range(h):
            r, g, b = image.getpixel((x, y))
            if r == 255 and g == 0 and b == 255:
                image.putpixel((x, y), (0, 0, 0))
            else:
                image.putpixel((x, y), (255, 255, 255))
    return image

# 定义类别数量
CLASS_NUM = 2
# 定义所有图片地址
true_mask_path = './data_road/training/gt_image_2/'
# 获取所有图片地址
true_mask_list = glob.glob(true_mask_path+'*.png')
# print(true_mask_list[0])
# 定义每一个类别的像素总量存放位置
classPixelCount = np.zeros(CLASS_NUM, dtype=np.float32)
# 定义含有某个类别像素的图片像素总量存放位置。
classTotalCount = np.zeros(CLASS_NUM, dtype=np.float32)
# print(classPixelCount, classTotalCount)
for image in true_mask_list:
    image_label = change_label(Image.open(image))
    image_label = np.array(image_label.convert('L')).flatten()
    perImageFrequencies = np.bincount(image_label)
    # print(perImageFrequencies)
    classPixelCount += [perImageFrequencies[0], perImageFrequencies[-1]]
    perImageFrequencies_class = [perImageFrequencies[0], perImageFrequencies[-1]]
    # print(classPixelCount)
    nPixelsInImage = perImageFrequencies[0] + perImageFrequencies[-1]
    for i in range(CLASS_NUM):
        if perImageFrequencies_class[i] > 0:
            classTotalCount[i] += nPixelsInImage
    # print(classTotalCount)
frequencies = np.divide(classPixelCount, classTotalCount)
print(frequencies)
知道了每个类别的权重之后，就该取它们的中位数，然后用每个权重取除以这个中位数
再用它除以每个类的频率。
# frequencies = [0.15314929, 0.8468501 ]
median = (frequencies[0] + frequencies[1]) / 2
weights = [median / x for x in frequencies]
print(weights)
# weights = [3.26478624223462, 0.5904229036520159]
