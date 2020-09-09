"""
@Author 张恒
CV核心课第一周作业
"""
# 显示设置: 只显示最终效果
showBestResultsOnly = False


def makeImages(images, figname, numOfCols=4):
    from matplotlib.pyplot import figure
    fig = figure(figname,figsize=(20,20))
    fig.suptitle(figname, y=1, fontsize=30)
    
    numOfImages = len(images)
    numOfRows = numOfImages // numOfCols
    if numOfImages % numOfCols != 0:
        numOfRows += 1
    imageIndex = 1
    for key in images.keys():
        ax = fig.add_subplot(numOfRows, numOfCols, imageIndex)
        ax.imshow(images[key])
        ax.set_title(key,fontsize=20,color=(1,0,0))
        ax.set_xticks([])
        ax.set_yticks([])
        imageIndex += 1
    return fig


def addWaterMark(img, index, mask):
    from cv2 import putText,FONT_HERSHEY_SIMPLEX,split,merge
    img = putText(img, index, (50, 150), FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
    imgR,imgG,imgB = split(img)
    maskB,maskG,maskR = split(mask)
    from numpy import where
    indexs = where(maskB==0)
    imgR[indexs[0],indexs[1]] = 255
    imgG[indexs[0], indexs[1]] = 255
    imgB[indexs[0], indexs[1]] = 255
    return merge((imgR,imgG,imgB))


# 读取原始图片
from cv2 import imread, cvtColor, COLOR_BGR2RGB

rawImage = imread("week1_homework.png")
rawImage = cvtColor(rawImage, COLOR_BGR2RGB)


"""
尝试自定义卷积核滤波器
"""
# 定义滤波结果集合
convFilterResImgs = {"Raw Image":rawImage}

import numpy as np
from cv2 import filter2D

for ksize in range(5, 50, 5):
    # 创建卷积核
    kernal = np.ones([ksize, ksize], dtype=np.float32) / (ksize ** 2)
    # 滤波操作
    resImage = filter2D(rawImage, -1, kernal)
    convFilterResImgs["ksize = " + str(ksize)] = resImage


"""
尝试常见线性滤波器
"""
# 定义滤波结果集合
linearFilterResImgs = {}

# 均值滤波
from cv2 import blur
for ksize in range(5, 24, 6):
    linearFilterResImgs["Mean Filter (ksize=" + str(ksize) + ")"] = blur(rawImage, (ksize, ksize))

# 高斯滤波
from cv2 import GaussianBlur
for ksize in range(5, 24, 6):
    linearFilterResImgs["Gaussian Filter (ksize=" + str(ksize) + ")"] = GaussianBlur(rawImage, (ksize, ksize), 1.5)

# 方框滤波
from cv2 import boxFilter
for ksize in range(5, 24, 6):
    linearFilterResImgs["Box Filter (ksize=" + str(ksize) + ")"] = boxFilter(rawImage, -1, (ksize, ksize))

"""
尝试常见非线性滤波器
"""
# 定义滤波结果集合
nonlinearFilterResImgs = {}

# 中值滤波
from cv2 import medianBlur
for ksize in range(9, 24, 2):
    nonlinearFilterResImgs["Median Filter (ksize=" + str(ksize) + ")"] = medianBlur(rawImage, ksize)

# 双边滤波
from cv2 import bilateralFilter
for ksize in range(9, 24, 2):
    nonlinearFilterResImgs["Bilateral Filter (ksize=" + str(ksize) + ")"] = bilateralFilter(rawImage, ksize, sigmaColor=2, sigmaSpace=2)


"""
显示所有的效果
"""
if not showBestResultsOnly:
    f1 = makeImages(convFilterResImgs, "Customized Convolutional Filters")
    f2 = makeImages(linearFilterResImgs, "Linear Filters")
    f3 = makeImages(nonlinearFilterResImgs, "Nonlinear Filters")
    f1.show()
    f2.show()
    f3.show()

"""
显示各种滤波器的最佳效果
"""
# 添加水印
watermark = imread("watermark.png")
bestResultsImages = {}
bestResultsImages["Customized Kernal"] = addWaterMark(convFilterResImgs["ksize = " + str(25)],"A",watermark)
bestResultsImages["Mean"] = addWaterMark(linearFilterResImgs["Mean Filter (ksize=" + str(23) + ")"],"B",watermark)
bestResultsImages["Gaussian"] = addWaterMark(linearFilterResImgs["Gaussian Filter (ksize=" + str(23) + ")"],"C",watermark)
bestResultsImages["Box"] = addWaterMark(linearFilterResImgs["Box Filter (ksize=" + str(23) + ")"],"D",watermark)
bestResultsImages["Median"] = addWaterMark(nonlinearFilterResImgs["Median Filter (ksize=" + str(19) + ")"],"E",watermark)
bestResultsImages["Bilateral"] = addWaterMark(nonlinearFilterResImgs["Bilateral Filter (ksize=" + str(23) + ")"],"F",watermark)
f = makeImages(bestResultsImages, "Best Results", numOfCols=3)
f.show()

"""
总结：
中值滤波（ksize=19时，见Best Results图E）效果最佳
"""
from matplotlib.pyplot import imshow,show
bestRes = addWaterMark(nonlinearFilterResImgs["Median Filter (ksize=" + str(19) + ")"],"Best",watermark)
imshow(bestRes)
show()
print("中值滤波（ksize=19时）效果最佳，见图Best Results中的 图E")