"""
    CV核心课作业 Week4
    张恒
    解题思路：
    1、数据集构建，自动扫描路径下的文件夹，并以文件夹名称作为分类标签，以文件夹内的图片文件作为训练数据，建立映射表（元数据）
    2、数据加载器，内部通过元数据访问数据集内的对应文件
    3、模型过程，采用输入层-隐藏层-输出层的3层结构 BPNN
    4、训练过程，采用梯度下降法，调整权值
    5、测试过程，采用简单交叉验证法，随机将70%数据分配为训练集，30%数据作分配为测试集

    结果表明：测试准确率达98%以上

    其他依赖的Python包：
    pandas
"""

""" 工具函数 """
# 扫描路径下所有文件夹和文件
def dir_and_file(path=None):
    if not path:
        from os.path import abspath
        path = abspath(".")
    from os import listdir
    from os.path import isfile,isdir,normpath
    pwd = normdir(path)
    pathlist = [normpath(pwd + pname) for pname in listdir(pwd)]
    # 过滤文件夹
    dirlist = [pname for pname in pathlist if isdir(pname)]
    # 过滤文件
    filelist = [pname for pname in pathlist if isfile(pname)]
    return dirlist,filelist

# 路径后面加"\"
def normdir(path):
    return path if path.endswith("\\") or path.endswith("/") else path + "\\"

# 扫描数据集获得元数据
def init_dataset(path):
    dirs, files = dir_and_file(path)
    datamapper = {}
    filelist = []
    labellist = []
    from os.path import basename
    for dir in dirs:
        _, filepaths = dir_and_file(dir)
        filelist.extend(filepaths)
        labellist.extend([basename(dir)] * len(filepaths))
    datamapper["file"] = filelist
    # 将文件夹名称转换为数字标签
    datamapper["label"] = digital_label(labellist)
    from pandas import DataFrame
    # 生成dataframe
    metadata = DataFrame(datamapper)
    print(metadata.head())
    return metadata

# 将名称转化为数字标签
def digital_label(labellist):
    ylabellist = []
    history = []
    for label in labellist:
        if label not in history:
            history.append(label)
        index = history.index(label)
        ylabellist.append(index)
    return ylabellist

# 将数据封装为数据加载器
def make_dataset(path,train=0.7,test=0.3):
    metadata = init_dataset(path)
    numtrain = round(metadata.shape[0] * train / (train + test))
    from numpy import zeros,ones,bool,hstack
    from numpy.random import permutation
    bools = permutation(hstack([ones(numtrain),zeros(metadata.shape[0] - numtrain)]).astype(bool))
    trainmetadata = metadata.iloc[bools]
    testmetadata = metadata.iloc[~bools]
    return MyDataset(trainmetadata,transform=transforms.ToTensor()),MyDataset(testmetadata,transform=transforms.ToTensor())

# 定义数据集类
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from cv2 import imread,cvtColor,COLOR_BGR2GRAY

class MyDataset(Dataset):
    def __init__(self, metadata, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.metadata = metadata
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filepath = self.metadata.iat[index,0]
        label = self.metadata.iat[index,1]
        image = imread(filepath)
        image = cvtColor(image,COLOR_BGR2GRAY)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.metadata.shape[0]

# 计算平均准确率
def cal_acc(ypred,yreal):
    acc = max(ypred, 1)[1].numpy() == yreal.numpy()
    return acc.mean()


'''
    创建数据加载器
'''
trainset,testset = make_dataset("./homework_dataset")
trainloader = DataLoader(dataset=trainset, batch_size=50, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=len(testset), shuffle=True)

'''
    创建神经网络
'''
from torch.nn import Module,Linear,Sequential
class BPNN(Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.layers = Sequential(
            Linear(30 * 30, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

model = BPNN()
print(model)

'''
    设置训练参数
'''
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
# 损失函数
loss_function = CrossEntropyLoss()
# 优化器
optimizer = Adam(model.parameters(), lr=0.001)
# 最大训练遍数
max_epoch = 20

'''
    训练并测试模型
'''
from torch.autograd import Variable
from torch import max
losses = []
trainaccs = []
testaccs = []
for epoch in range(max_epoch):
    for index, (x,y) in enumerate(trainloader):
        bacth_x = Variable(x)
        bacth_y = Variable(y)
        # 梯度置零
        optimizer.zero_grad()
        # 正向传播
        ypred = model(bacth_x)
        # 计算损失
        lossval = loss_function(ypred, bacth_y)
        # 反向传播
        lossval.backward()
        # 调整权值
        optimizer.step()
        # 每5次迭代输出一次损失值和测试准确率
        if not index % 5:
            # 记录损失值
            losses.append(lossval)
            # 记录训练准确率
            trainacc = cal_acc(ypred, bacth_y)
            trainaccs.append(trainacc)
            # 测试数据
            tx, ty = iter(testloader).next()
            test_x = Variable(tx)
            test_y = Variable(ty)
            typred = model(test_x)
            testacc = cal_acc(typred, test_y)
            testaccs.append(testacc)
            print(f"损失值 = {lossval.item()} , 训练准确率 = {trainacc}, 测试准确率 = {testacc}")

'''
    作图
'''
from matplotlib.pyplot import show,figure,legend
fig = figure()
ax1 = fig.add_subplot(111)
ax1.plot(losses,"b",label="training loss")
ax1.set_ylabel("loss")
ax2 = ax1.twinx()
ax2.plot(trainaccs,"g",label="training acc")
ax2.plot(testaccs,"r",label="testing acc")
ax2.set_ylabel("acc")
ax2.set_xlabel("iteration")
legend(loc='upper right')
show()