import torch
#------------------------IOU计算-------------------------------
#假设box格式如(x1,y1,x2,y2)其中x1<x2,y1<y2，box1是A个box的集合，box2是B个box的集合
def IOU(self,box1,box2):
    #找出两个框相交部分左上角的点
    A = box1.size(0)
    B = box2.size(0)
    # box1和box2最多有N*M个相交框，[N,2]->[N,1,2]->[N,M,2]，[M,2]->[1,M,2]->[N,M,2]
    max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box2[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(A, B, 2),
                       box2[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
#------------------------nms极大值抑制-------------------------------

def nms(self, bboxes, scores, threshold=0.5):
    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # 每个bbox的面积，维度[N,]
    areas = (x2 - x1) * (y2 - y1)
    #降序排列,得到其索引作为排名order
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的重叠面积
        xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]
        #交并比iou计算
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        #idx为与保留框差异大于阈值的索引列表
        idx = (iou <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        # 修补索引之间的差值
        order = order[idx + 1]
    # Pytorch的索引值为LongTensor
    return torch.LongTensor(keep)