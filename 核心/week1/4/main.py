# coding:utf-8
import cv2
import numpy as np
from side_window_filter import SideWindowFiltering_3d


# 显示图片
def cv_show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 制作水印
def make_watermark(height, width):
    image = 255*np.ones((height, width, 3), dtype=np.uint8)
    cv2.putText(image, 'Yuhua', (int(width / 2) - 100, height - 300), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
    #cv_show(image, 'Watermark')
    cv2.imwrite('Watermark.png', image)
    return image


if __name__ == '__main__':
    # 读取图片
    img = cv2.imread('week1_homework.png')

    # 制作水印
    height, width, _ = img.shape
    watermark = make_watermark(height, width)

    # 应用基于SideWindowFilter的中值滤波，保边效果好，运行时间较长，预计3分钟
    import time
    start = time.time()
    img_swf = SideWindowFiltering_3d(img, kernel=21, mode='median')
    end = time.time()
    print("It consumes %s." % (end-start))

    #应用均值、中值滤波、高斯滤波、双边滤波
    img_mean = cv2.blur(img, (7,7))  # kernel越大，图片越模糊，效果不明显
    img_median = cv2.medianBlur(img, 21) # kernel越大，去噪效果越好，但图片也越模糊
    img_gaussian = cv2.GaussianBlur(img, (5,5), 1)  # 无论kernel多大，效果不明显
    img_bilateral = cv2.bilateralFilter(img, 21, 42, 10.5)  # kernel越大，去噪效果越好，但图片也越模糊

    img_list = [img_swf, img_mean, img_median, img_gaussian, img_bilateral]
    img_name_list = ['img_swf.png', 'img_mean.png', 'img_median.png', 'img_gaussian.png', 'img_bilateral.png']
    # 添加水印并保存图片
    for i in range(len(img_list)):
        img0 = img_list[i]
        img0 = cv2.addWeighted(img0, 0.8, watermark, 0.2, 0)
        cv2.imwrite(img_name_list[i], img0)