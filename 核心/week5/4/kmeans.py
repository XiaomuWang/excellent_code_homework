#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/15 19:50

# @Author : ZFJ

# @File : kmeans.py 

# @Software: PyCharm
"""

import torch


# 欧式距离
def l2_distance(point1, point2):
    return torch.sqrt(torch.sum((point1.float() - point2.float()) ** 2))


# 求中心点
def get_center_point(points):
    center_point = torch.zeros_like(points[0], dtype=torch.float32)
    for point in points:
        center_point = center_point + point
    return center_point / len(points)


# 拿到离point最远的点
def get_farthest_point(point, compare_points):
    max_distance = torch.finfo(torch.float32).min
    farthest_point = None
    index = -1
    for i, compare_point in enumerate(compare_points):
        distance = l2_distance(point, compare_point)
        if distance > max_distance:
            max_distance = distance
            farthest_point = compare_point
            index = i
    return index, farthest_point


# 拿到离point最近的点
def get_nearest_point(point, compare_points):
    min_distance = torch.finfo(torch.float32).max
    nearest_point = None
    index = -1
    for i, compare_point in enumerate(compare_points):
        distance = l2_distance(point, compare_point)
        if distance < min_distance:
            min_distance = distance
            nearest_point = compare_point
            index = i
    return index, nearest_point


# 这个类用来把point包装为dict的key
class PointKey(object):
    def __init__(self, point):
        self.point = point
        # 坐标的每个数字用_来拼接
        self.name = '_'.join(list(map(lambda float_value: str(float_value), point.reshape(-1))))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name


class Kmeans(object):
    def __init__(self, k, points=None):
        self.k = k
        self.points = points
        self.clusters = None

    def init_clusters(self):
        # 最开始无脑选择前k个点作为聚类中心点, 也可以改成随机选k个点
        clusters = {}
        for i in range(self.k):
            clusters[PointKey(self.points[i])] = []
        self.clusters = clusters

    # 已拿到各个聚类中心点，遍历所有点来归类到那几个中心点
    def cluster(self):
        for point in self.points:
            # 求当前点离哪个中心点最近，该点就属于中心点这一类
            _, nearest_cluster_point = get_nearest_point(point,
                                                         list(map(lambda point_key: point_key.point, self.clusters)))
            self.clusters[PointKey(nearest_cluster_point)].append(point)
        return self.clusters

    def step(self, next_points=None):
        if next_points is not None:
            self.points = next_points

        # 还没初始化中心点就初始化
        if self.clusters is None:
            self.init_clusters()
            # 初始化完了之后，对所有点聚类
            return self.cluster()
        # 已经初始化过中心点
        else:
            new_cluster = {}
            # 对每类内部再求新的中心点
            for cluster_point, points_of_one_cluster in self.clusters.items():
                if len(points_of_one_cluster) == 0:
                    new_cluster[cluster_point] = []
                else:
                    new_cluster[PointKey(get_center_point(points_of_one_cluster))] = []
            self.clusters = new_cluster
            # 根据新的中心点，再遍历所有点进行聚类
            return self.cluster()

    def getClusterType(self, point):
        index, _ = get_nearest_point(point, list(map(lambda point_key: point_key.point, list(self.clusters.keys()))))
        return index


if __name__ == '__main__':
    points = []
    for i in range(100):
        points.append(torch.rand(2) * 100)

    kmeans = Kmeans(5, points)
    cluster_points = None
    for i in range(100):
        cluster_points = kmeans.step()

    print(cluster_points)
