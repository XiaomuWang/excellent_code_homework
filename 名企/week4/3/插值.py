import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

## 1.自己实现最近邻插值，双线性插值，双三次插值，并和opencv的比较

'''
说明：利用python/numpy/opencv实现图像插值法（最邻近，双线性，双三次(Bell分布)）
算法思路:
        1)以彩色图的方式加载图片;
        2)根据想要生成的图像大小，映射获取某个像素点在原始图像中的浮点数坐标;
		3)根据浮点数坐标确定插值算法中的系数、参数；
		4)采用不同的算法实现图像插值。

插值法的第一步都是相同的，计算目标图（dstImage）的坐标点对应原图（srcImage）中哪个坐标点来填充，计算公式为：
srcX = dstX * (srcWidth/dstWidth)
srcY = dstY * (srcHeight/dstHeight)
(dstX,dstY)表示目标图像的某个坐标点，(srcX,srcY)表示与之对应的原图像的坐标点。srcWidth/dstWidth 和 srcHeight/dstHeight 分别表示宽和高的放缩比。
那么问题来了，通过这个公式算出来的 srcX, scrY 有可能是小数，但是原图像坐标点是不存在小数的，都是整数，得想办法把它转换成整数才行。
不同插值法的区别就体现在 srcX, scrY 是小数时，怎么将其变成整数去取原图像中的像素值。
最近邻插值（Nearest-neighborInterpolation）：看名字就很直白，四舍五入选取最接近的整数。这样的做法会导致像素变化不连续，在目标图像中产生锯齿边缘。
双线性插值（Bilinear Interpolation）：双线性就是利用与坐标轴平行的两条直线去把小数坐标分解到相邻的四个整数坐标点。权重与距离成反比。
双三次插值（Bicubic Interpolation）：与双线性插值类似，只不过用了相邻的16个点。但是需要注意的是，前面两种方法能保证两个方向的坐标权重和为1，但是双三次插值不能保证这点，所以可能出现像素值越界的情况，需要截断。
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Nearest( img, bigger_height, bigger_width, channels  ):
    '''
    各种插值算法在图像领域的应用一般是用来将W * H像素大小的图片转化成(W * radio1) * (H * radio2)像素大小（radio1，radio2 > 1）的图片。
    最近邻插值算法顾名思义：用离该像素点位置最近的像素点的值来填充该像素点。
    想要插值，首先必须要有空地方给你插，因此插值的第一步是将原图像上的所有目标点转移到新图像上的对应位置。
    于是，我们设计的函数中，输入参数除了原图像之外，还要有W和H方向上的放大比例ratio。
    然而用户在调用的时候往往只是想将图片转化成一个固定大小的图片，他根本不关心原图的大小和什么破比例。
    没事儿作为懂事儿的程序我们得自己算。
    ratio_H = tar_H/src_H = tar_x/src_x
    ratio_W = tar_W/src_W = tar_y/src/y
    目标图像的像素点和原始图像的像素点映射如下:
    tar_x = src_x * ratio_H
    tar_y = src_y * ratio_W
    最后我们再盘一下这个算法的流程：
    1）根据tar_H和tar_W创建目标图像画布
    2）计算比例缩放因子ratio
    3）遍历目标图像的每个像素点，计算映射关系
    4）遍历目标图像的每个像素点，使用对应原始图的对应像素点对其赋值。
    其中再具体实现的过程中，最难的应该是最后一步：
    原始图像上的那些点，可以直接将他放到对应映射关系的像素点上去。
    那在原始图像中没有映射点的目标图像上的点怎么办呢？
    这个算法给出的答案是找距离最近的那个点！
    怎么找？
    利用目标图片的像素索引值/映射比例，再取整即可
    '''
    near_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            near_row =  round ( row )
            near_col = round( col )
            if near_row == img.shape[0] or near_col == img.shape[1]:
                near_row -= 1
                near_col -= 1
                
            near_img[i][j] = img[near_row][near_col]
            
    return near_img

def Bilinear( img, bigger_height, bigger_width, channels ):
    bilinear_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            row_int = int( row )
            col_int = int( col )
            u = row - row_int
            v = col - col_int
            if row_int == img.shape[0]-1 or col_int == img.shape[1]-1:
                row_int -= 1
                col_int -= 1
                
            bilinear_img[i][j] = (1-u)*(1-v) *img[row_int][col_int] + (1-u)*v*img[row_int][col_int+1] + u*(1-v)*img[row_int+1][col_int] + u*v*img[row_int+1][col_int+1]
            
    return bilinear_img

def Bicubic_Bell( num ):
   # print( num)
    if  -1.5 <= num <= -0.5:
      #  print( -0.5 * ( num + 1.5) ** 2 )
        return -0.5 * ( num + 1.5) ** 2
    if -0.5 < num <= 0.5:
       # print( 3/4 - num ** 2 )
        return 3/4 - num ** 2
    if 0.5 < num <= 1.5:
       # print( 0.5 * ( num - 1.5 ) ** 2 )
        return 0.5 * ( num - 1.5 ) ** 2
    else:
       # print( 0 )
        return 0
        
    
def Bicubic ( img, bigger_height, bigger_width, channels ):
    Bicubic_img = np.zeros( shape = ( bigger_height, bigger_width, channels ), dtype = np.uint8 )
    
    for i in range( 0, bigger_height ):
        for j in range( 0, bigger_width ):
            row = ( i / bigger_height ) * img.shape[0]
            col = ( j / bigger_width ) * img.shape[1]
            row_int = int( row )
            col_int = int( col )
            u = row - row_int
            v = col - col_int
            tmp = 0
            for m in range( -1, 3):
                for n in range( -1, 3 ):
                    if ( row_int + m ) < 0 or (col_int+n) < 0 or ( row_int + m ) >= img.shape[0] or (col_int+n) >= img.shape[1]:
                        row_int = img.shape[0] - 1 - m
                        col_int = img.shape[1] - 1 - n

                    numm = img[row_int + m][col_int+n] * Bicubic_Bell( m-u ) * Bicubic_Bell( n-v ) 
                    tmp += np.abs( np.trunc( numm ) )
                    
            Bicubic_img[i][j] = tmp
    return Bicubic_img

# 首先运行一下opencv自带的插值函数
img = cv2.imread( './data_road/training/image_2/um_000000.png',  cv2.IMREAD_COLOR)
img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
print( img[3][3] )
height, width, channels = img.shape
print( height, width )

bigger_height = height + 200
bigger_width = width + 200
print( bigger_height, bigger_width)
    
near_img = cv2.resize(img, dsize=(bigger_width, bigger_height), interpolation = cv2.INTER_NEAREST)
bilinear_img = cv2.resize(img, dsize=(bigger_width, bigger_height), interpolation = cv2.INTER_LINEAR)
Bicubic_img = cv2.resize(img, dsize=(bigger_width, bigger_height), interpolation = cv2.INTER_CUBIC)
    
plt.figure(figsize=(50, 50))
plt.subplot( 2, 2, 1 )
plt.title( 'Source_Image' )
plt.imshow( img ) 
plt.subplot( 2, 2, 2 )
plt.title( 'Nearest_Image' )
plt.imshow( near_img )
plt.subplot( 2, 2, 3 )
plt.title( 'Bilinear_Image' )
plt.imshow( bilinear_img )
plt.subplot( 2, 2, 4 )
plt.title( 'Bicubic_Image' )
plt.imshow( Bicubic_img )
plt.show()

# 再用自己的方法实现一遍
# 咱也不知道为啥，这opencv运行的吧，
# 比咱快那么多，估计是人家优化做的好吧。。。有空去看看人家怎么优化的。
img = cv2.imread( './data_road/training/image_2/um_000000.png',  cv2.IMREAD_COLOR)
img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
print( img[3][3] )
height, width, channels = img.shape
print( height, width )

bigger_height = height + 200
bigger_width = width + 200
print( bigger_height, bigger_width)
    
near_img = Nearest( img, bigger_height, bigger_width, channels )
bilinear_img = Bilinear( img, bigger_height, bigger_width, channels )
Bicubic_img = Bicubic( img, bigger_height, bigger_width, channels )
    
plt.figure(figsize=((bigger_height * 2) // 10, (bigger_width * 2) // 10))
plt.subplot( 2, 2, 1 )
plt.title( 'Source_Image' )
plt.imshow( img ) 
plt.subplot( 2, 2, 2 )
plt.title( 'Opencv_Nearest_Image' )
plt.imshow( near_img )
plt.subplot( 2, 2, 3 )
plt.title( 'Opencv_Bilinear_Image' )
plt.imshow( bilinear_img )
plt.subplot( 2, 2, 4 )
plt.title( 'Opencv_Bicubic_Image' )
plt.imshow( Bicubic_img )
plt.show()
