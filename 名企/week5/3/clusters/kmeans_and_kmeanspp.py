# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/9/27

from pylab import *
from numpy import *
import codecs
import matplotlib.pyplot as plt


def distance(x1, x2):
    """
    计算两个样本间的欧式距离
    """
    return sqrt(sum(power(x1 - x2, 2)))


def nearest(point, cluster_centers):
    """
    对一个样本找到最近的距离
    找到与该样本距离最近的聚类中心
    """
    min_dist = inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def randcenter(set, k):
    """
    随机初始化聚类中心
    适用于kmeans算法初始化聚类中心
    """
    dim = shape(set)[1]
    init_cen = zeros((k, dim))
    for i in range(dim):
        min_i = min(set[:, i])
        range_i = float(max(set[:, i]) - min_i)
        init_cen[:, i] = min_i + range_i * random.rand(k)
    return init_cen


def get_centroids(dataset, k):
    """
    选择尽可能相距较远的初始聚类中心
    适用于kmeans++算法初始化聚类中心, 轮盘赌依次选择
    """
    m, n = np.shape(dataset)    # m:样本数 n:数据维度
    cluster_centers = np.zeros((k, n))

    # 1、 随机选一个初始聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = dataset[index, ]
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataset[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.rand()
        # 6、获得距离最远的样本点作为聚类中心点
        # 轮盘赌选择
        for j, di in enumerate(d):
            sum_all = sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i, ] = dataset[j, ]
            break
    return cluster_centers


def Kmeans(dataset, k):
    """
    kmeans 算法主程序
    :param dataset: 数据集
    :param k: 聚类数
    """
    row_m = shape(dataset)[0]
    cluster_assign = zeros((row_m, 2))
    center = randcenter(dataset, k)
    change = True
    while change:
        change = False
        for i in range(row_m):
            mindist = inf
            min_index = -1
            for j in range(k):
                distance1 = distance(center[j, :], dataset[i, :])
                if distance1 < mindist:
                    mindist = distance1
                    min_index = j
            if cluster_assign[i, 0] != min_index:
                change = True
            cluster_assign[i, :] = min_index, mindist ** 2
        for cen in range(k):
            cluster_data = dataset[nonzero(cluster_assign[:, 0] == cen)]
            center[cen, :] = mean(cluster_data, 0)
    return center, cluster_assign


def KmeansPlus(dataset, k):
    """
    kmeans++ 算法主程序
    :param dataset: 数据集
    :param k: 聚类数
    """
    row_m = shape(dataset)[0]
    cluster_assign = zeros((row_m, 2))
    center = get_centroids(dataset, k)
    change = True
    while change:
        change = False
        for i in range(row_m):
            mindist = inf
            min_index = -1
            for j in range(k):
                distance1 = distance(center[j, :], dataset[i, :])
                if distance1 < mindist:
                    mindist = distance1
                    min_index = j
            if cluster_assign[i, 0] != min_index:
                change = True
            cluster_assign[i, :] = min_index, mindist ** 2
        for cen in range(k):
            cluster_data = dataset[nonzero(cluster_assign[:, 0] == cen)]
            center[cen, :] = mean(cluster_data, 0)
    return center, cluster_assign


if __name__ == '__main__':

    # 初始化数据及标签以及聚类数
    data = []
    labels = []
    k = 5

    # 数据读取
    with codecs.open("./data1.txt", "r") as f:
        for line in f.readlines():
            # 每一行数据为 x, y, label
            x, y, label = line.strip().split('\t')
            data.append([float(x), float(y)])
            labels.append(float(label))
    datas = array(data)

    # --------------kmeans++--------------
    cluster_center, cluster_assign = KmeansPlus(datas, k)
    print(cluster_center)
    # 设置x,y轴的范围
    xlim(-10, 20)
    ylim(-10, 20)
    # 做散点图
    plt.title('Kmeans++')
    f1 = plt.figure(1)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 0), 0], datas[nonzero(cluster_assign[:, 0] == 0), 1], marker='o',
                color='r', label='0', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 1), 0], datas[nonzero(cluster_assign[:, 0] == 1), 1], marker='+',
                color='b', label='1', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 2), 0], datas[nonzero(cluster_assign[:, 0] == 2), 1], marker='*',
                color='g', label='2', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 3), 0], datas[nonzero(cluster_assign[:, 0] == 3), 1], marker='v',
                color='y', label='3', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 4), 0], datas[nonzero(cluster_assign[:, 0] == 4), 1], marker='^',
                color='k', label='4', s=30)
    plt.scatter(cluster_center[:, 1], cluster_center[:, 0], marker='x', color='m', s=50)
    plt.savefig("Kmeans++.jpg")
    plt.show()

    # --------------kmeans--------------
    cluster_center, cluster_assign = Kmeans(datas, k)
    print(cluster_center)
    # 设置x,y轴的范围
    xlim(-10, 20)
    ylim(-10, 20)
    # 做散点图
    plt.title('Kmeans')
    f1 = plt.figure(1)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 0), 0], datas[nonzero(cluster_assign[:, 0] == 0), 1], marker='o',
                color='r', label='0', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 1), 0], datas[nonzero(cluster_assign[:, 0] == 1), 1], marker='+',
                color='b', label='1', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 2), 0], datas[nonzero(cluster_assign[:, 0] == 2), 1], marker='*',
                color='g', label='2', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 3), 0], datas[nonzero(cluster_assign[:, 0] == 3), 1], marker='v',
                color='y', label='3', s=30)
    plt.scatter(datas[nonzero(cluster_assign[:, 0] == 4), 0], datas[nonzero(cluster_assign[:, 0] == 4), 1], marker='^',
                color='k', label='4', s=30)
    plt.scatter(cluster_center[:, 1], cluster_center[:, 0], marker='x', color='m', s=50)
    plt.savefig("Kmeans.jpg")
    plt.show()
