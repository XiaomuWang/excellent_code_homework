"""
本模块主要是构建Lenet模型
"""
import torch.nn as nn


class LeNet(nn.Module):
    """
    Input - 1x28x28
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, [5, 5], stride=1)
        self.pool1 = nn.MaxPool2d([2, 2])
        self.conv2 = nn.Conv2d(6, 16, [5, 5], stride=1)
        self.pool2 = nn.MaxPool2d([2, 2])
        self.conv3 = nn.Conv2d(16, 120, [5, 5], stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img):
        img = self.conv1(img)
        img = self.pool1(img)
        img = self.conv2(img)
        img = self.pool2(img)
        img = self.conv3(img)
        img = img.view(-1, 120)
        img = self.fc1(img)
        img = self.dropout(img)
        img = self.fc2(img)
        return img


def lenet():
    return LeNet()


if __name__ == "__main__":
    import torch
    img = torch.randn([1, 1, 32, 32])
    net = LeNet()
    print(net(img))
