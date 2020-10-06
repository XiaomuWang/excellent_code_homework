import torch
from torch import nn

# 二分类focal loss
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pt, target):
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
          (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss