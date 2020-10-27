# coding:utf-8

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 16
DATA_PATH = './data'
PRINT_NUM = 2000
PRINT_NUM_MOD = PRINT_NUM - 1

optimizer_names = ['Momentum', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
                                # transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(optimizer_name, epoch, net, optimizer, criterion):
    net.train()
    total_losses = 0.0
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        total_losses += loss_val
        running_loss += loss_val
        if i % PRINT_NUM == PRINT_NUM_MOD:
            print('[%s, %d, %5d] loss: %.3f' % (optimizer_name, epoch + 1, i + 1, running_loss / (BATCH_SIZE * PRINT_NUM)))
            running_loss = 0.0
    return running_loss / len(train_loader.dataset)


def evaluate(net):
    net.eval()
    correct = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


def draw_figure(data, title="Loss", ylabel='loss', filename="loss.png"):
    plt.clf()
    for i, d in enumerate(data):
        plt.plot(d, label=optimizer_names[i])
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)


if __name__ == '__main__':
    train_set = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    nets = [Net().to(device) for _ in optimizer_names]
    optimizers = [
        optim.SGD(nets[0].parameters(), lr=0.01, momentum=0.8),
        optim.Adagrad(nets[1].parameters(), lr=0.01),
        optim.RMSprop(nets[2].parameters(), lr=0.001, alpha=0.99),
        optim.Adadelta(nets[3].parameters(), lr=0.03),
        optim.Adam(nets[4].parameters(), lr=0.001, betas=(0.9, 0.99))
    ]

    losses = [[], [], [], [], []]
    top1_accs = [[], [], [], [], []]
    for i, (optimizer_name, net, optimizer) in enumerate(zip(optimizer_names, nets, optimizers)):
        for epoch in range(EPOCH):
            loss = train(optimizer_name, epoch, net, optimizer, criterion)
            losses[i].append(loss)
            top1_acc = evaluate(net)
            top1_accs[i].append(top1_acc)

    draw_figure(losses, title="Loss", ylabel='loss', filename="loss.png")
    draw_figure(top1_accs, title="Top1-Acc", ylabel='top1-acc', filename="top1_acc.png")
    print("运行完成！！！")
