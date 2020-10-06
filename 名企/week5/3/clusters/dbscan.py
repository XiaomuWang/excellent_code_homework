# -*- coding: utf-8 -*- #
# Author: Henry
# Date:   2020/9/27

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import DBSCAN


if __name__ == '__main__':

    # 加载数据
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
    print(X.shape)

    # 绘制数据分布图
    plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()

    # 设定聚类并拟合数据
    dbscan = DBSCAN(eps=0.4, min_samples=3)
    dbscan.fit(X)
    label_pred = dbscan.labels_

    # 绘制dbscan聚类结果
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title("DBSCAN Clustering")
    plt.legend(loc=2)
    plt.show()
