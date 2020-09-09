# coding:utf-8
import cv2
import numpy as np


def generate_data():
    # 本函数生成0-9，10个数字的图片矩阵
    image_data = []
    num_0 = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_0)
    num_1 = np.array(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_1)
    num_2 = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_2)
    num_3 = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_3)
    num_4 = np.array(
        [[0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_4)
    num_5 = np.array(
        [[0, 1, 1, 1, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_5)
    num_6 = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_6)
    num_7 = np.array(
        [[0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_7)
    num_8 = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_8)
    num_9 = np.array(
        [[0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_9)
    return image_data


def generate_test_data():
    test_img = []
    # 构造数字0，图像尺寸大小6x6
    test_zero = np.array(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0]])
    test_img.append(test_zero)

    # 构造数字1，图像尺寸大小7x7
    test_one = np.array(
        [[0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    )
    test_img.append(test_one)

    # 构造数字2，图像尺寸大小8x8
    test_two = num_2 = np.array(
        [[0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 0]])
    test_img.append(test_two)

     # 构造数字7， 图像尺寸大小为8x8
    test_seven = np.array(
        [[0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0]])
    test_img.append(test_seven)

    return test_img


def get_feature(image):
    # 定义hog描述子的参数
    cell_size = (1, 1)  # Cell size in pixels
    num_cells_per_block = (2, 2)  # Number of cells per block in each direction (x, y)
    block_size = (num_cells_per_block[0]*cell_size[0],
                  num_cells_per_block[1]*cell_size[1])   # Block size in pixels
    x_cells = image.shape[1]
    y_cells = image.shape[0]
    h_stride = 2  # Horizontal distance between blocks in units of Cell Size
    v_stride = 2  # Vertical distance between blocks in units of Cell Size
    block_stride = (cell_size[0]*h_stride, cell_size[1]*v_stride)  # Block stride in pixels (horizontal, vertical)
    num_bins = 7  # Number of bins for the histograms
    win_size = (x_cells*cell_size[0], y_cells*cell_size[1])  # Size of detection window in pixels

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    # 使用HOG算法提取x的直方图特征描述子
    hog_feature = hog.compute(image)
    return hog_feature


def bin2gray(binary_img):
    # 将二值化的图像转换为灰度图
    gray_img = (binary_img * 255).astype(np.uint8)
    gray_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_LINEAR)
    return gray_img


def cos_sim(vector_a, vector_b):
    # 计算两个向量之间的余弦相似度
    num = np.dot(vector_a.reshape(-1), vector_b.reshape(-1))
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def model(img):
    y = -1
    # 下面添加对feature进行决策的代码，判定出feature 属于[0,1,2,3,...9]哪个类别
    gray = bin2gray(img)
    _feature = get_feature(gray)
    y = np.argmax([cos_sim(_feature, x) for x in template])
    return y


if __name__ == "__main__":
    orignal_image_data = generate_data()
    # 打印出0的图像
    print("数字0对应的图片是:")
    print(orignal_image_data[0])
    print("-" * 20)

    # 打印出8的图像
    print("数字8对应的图片是:")
    print(orignal_image_data[8])
    print("-" * 20)

    # 对每张图片提取特征并组成模板库
    print("开始提取特征并组成模板库")
    template = []
    for i in range(0, 10):
        orignal_image = orignal_image_data[i]
        # 二值化图像转换为灰度图
        gray_orignal_image = bin2gray(orignal_image)
        # 保存图片
        # img_path = "./number_{}.png".format(i)
        # cv2.imwrite(img_path, gray_orignal_image)
        # 对当前图片提取特征
        feature = get_feature(gray_orignal_image)
        template.append(feature)
    print("成功完成模板库组建")
    print("-" * 20)
    print("生成测试数据库")
    test_img_data = generate_test_data()
    print("开始进行数字预测")
    for i in range(len(test_img_data)):
        print("第%s个测试图片是:" % (i+1))
        print(test_img_data[i])
        # 预测图片
        result = model(test_img_data[i])
        print("第%s个测试图片的预测结果是:%s" % (i+1, result))
        # 保存图片
        # img_path = "./test_img_{}.png".format(result)
        # cv2.imwrite(img_path, bin2gray(test_img_data[i]))
        print("-" * 20)

