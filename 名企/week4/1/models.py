# lanhuajian

from torch import nn
from torchvision.models import VGG


def buildVgg11FeatureLayers():
    return nn.Sequential(
        # 输入rgb三通道
        # 以下是vgg11的配置，写死的
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 2倍下采样

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 4倍下采样

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 8倍下采样

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 16倍下采样

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 32倍下采样
    )


# 这里的VGG父类是torchvision包中提供的父类
class Vgg11(VGG):
    def __init__(self):
        # vgg主要有两个部分：features和classifier，前者用来提取特征，后者用来分类
        # 必须要名称一样、结构一样，这样才能加载torch的vgg预训练参数
        super(Vgg11, self).__init__(features=buildVgg11FeatureLayers())

class Fcn(nn.Module):
    def __init__(self, scale, featureProxyNet, classesNum=2):
        super(Fcn, self).__init__()
        # 尺度，例如fcn8、fcn32
        self.scale = scale
        # 下采样的特征网络
        self.featureProxyNet = featureProxyNet
        # 分类数
        self.classesNum = classesNum
        # 获取所有下采样层，并倒转过来
        self.poolingLayers = list(reversed(self.getLayers(featureProxyNet.layers, 'Pool')))
        # 获取最后一个卷积层的输出channel数
        out_channels = self.getLayers(featureProxyNet.layers, 'Conv2d')[-1].out_channels
        # 当前特征网络做完下采样后，尺度变为原图的 1 / maxScale
        self.maxScale = 2 ** len(self.poolingLayers)
        in_channels = out_channels
        layers = []
        for _ in enumerate(self.poolingLayers):
            # 对每个下采样层构建对应的转置卷积和激活函数等，用来做上采样恢复到原图尺寸
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            # 每执行一次转置卷积，channel数下降一半
            out_channels //= 2

        # 最后还需要一个卷积层来做分类（语义分割），在本次作业中分为道路和背景两类，所以最终有两个channel
        layers.append(nn.Conv2d(out_channels*2, classesNum, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def getLayers(self, layers, name):
        # 找到包含某个名字的层
        return list(filter(lambda layer: name in layer.__class__.__name__, layers))

    def forward(self, input):
        # 用特征网络做下采样，得到每一层的输出结果
        featureOutputs = self.featureProxyNet(input)
        # 拿到最后一层的结果
        output = featureOutputs[-1]['result']
        # 拿到所有池化层（下采样）的结果，并倒序一下，之后用来与转置卷积结果进行相加
        # 倒序后变成 [1/32x, 1/16x, 1/8x, 1/4x, 1/2x]
        poolingOutputs = list(reversed(list(filter(lambda output: 'Pool' in output['layer'].__class__.__name__, featureOutputs))))
        # 从1/16x开始做转置卷积
        poolingOutputIndex = 1

        currentScale = self.maxScale
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d) and currentScale >= self.scale:
                # bn执行前先把下采样结果和上采样结果相加
                output = output + poolingOutputs[poolingOutputIndex]['result']
                poolingOutputIndex += 1
            elif isinstance(layer, nn.ConvTranspose2d):
                # 转置卷积运行一次，尺度增加2倍
                currentScale //= 2
            output = layer(output)

        return output


class ProxyNet(nn.Module):
    # 提供一系列的层
    def __init__(self, name, layers):
        super(ProxyNet, self).__init__()
        self.name = name
        self.layers = layers

    # 前向传播记录每一层的结果
    def forward(self, input):
        results = []
        layerInput = input
        for i in range(len(self.layers)):
            layerInput = self.layers[i](layerInput)
            results.append({'layer': self.layers[i], 'result': layerInput})
        return results