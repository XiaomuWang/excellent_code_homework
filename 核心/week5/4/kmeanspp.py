#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/15 19:16 

# @Author : ZFJ

# @File : kmeanspp.py

# @Software: PyCharm
"""

from kmeans import *

class Kmeanspp(Kmeans):
    # 重新初始化聚类中心点的方法，其他地方算法不变
    def init_clusters(self):
        # 最开始选第一个点作为第一个分类的中心点
        cluster_points = [self.points[0]]
        # 排除第一个中心点，得到剩余的点
        remain_points = self.points[1:]
        cluster = {}
        cluster[PointKey(cluster_points[0])] = []
        # 第1类的中心已经分好了，我们继续找剩余的k-1个聚类中心点
        for i in range(1, self.k):
            # 找到离当前所有中心点最远的点，作为新的一类
            center_point = get_center_point(cluster_points)
            index, new_cluster_point = get_farthest_point(center_point, remain_points)
            cluster[PointKey(new_cluster_point)] = []
            # 排除新的中心点，得到剩余的点
            remain_points.pop(index)
            cluster_points.append(new_cluster_point)
            if len(remain_points) == 0:
                break
        self.clusters = cluster


if __name__ == '__main__':
    points = []
    for i in range(100):
        points.append(torch.rand(2) * 100)

    kmeans = Kmeanspp(5, points)
    cluster_points = None
    for i in range(100):
        cluster_points = kmeans.step()

    print(cluster_points)