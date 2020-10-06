# lanhuajian

import torch


# 求交集数量
def intersectionCount(predict, gt):
    intersection = torch.zeros_like(predict)
    intersection[(predict == 1) & (gt == 1)] = 1
    return torch.sum(intersection, dim=1)


# 求并集数量
def unionCount(predict, gt):
    union = torch.zeros_like(predict)
    union[(predict == 1) | (gt == 1)] = 1
    return torch.sum(union, dim=1)


# 平均iou
def meanIou(predict, gt):
    # batch channel height weight
    B, C, H, W = predict.shape
    # 0 channel的值代表是否是道路, 1 channel的值代表是否是背景
    # 把channel展平，方便计数
    road_channel = predict[:, 0, :, :].reshape(B, H*W)
    background_channel = predict[:, 1, :, :].reshape(B, H*W)
    road = torch.zeros_like(road_channel)
    # 道路概率大于等于背景概率的，算作道路
    road[road_channel >= background_channel] = 1

    road_gt = gt[:, 0].reshape(B, H*W)
    road_intersection = intersectionCount(road, road_gt)
    road_union = unionCount(road, road_gt)

    background = torch.zeros_like(road_channel)
    # 道路概率小于等于背景概率的，算作背景
    background[road_channel <= background_channel] = 1
    background_gt = torch.zeros_like(road_gt)
    # 这里背景简单计算为 不是道路就是背景
    background_gt[road_gt == 0] = 1
    background_intersection = intersectionCount(background, background_gt)
    background_union = unionCount(background, background_gt)

    # iou计算公式: 交集 / (并集 - 交集)
    road_ious = road_intersection / (road_union - road_intersection)
    backgound_ious = background_intersection / (background_union - background_intersection)

    # 平均iou
    return torch.sum(road_ious / backgound_ious).item() / B
