import torch
import torch.nn as nn
import math
from collections import OrderedDict

# 基本的darknet块
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        '''
        1*1卷积后再3*3卷积是为了减少参数。1*1卷积后通道数会下降，3*3后通道数又会上升
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False) # 1*1卷积32通道 下降通道数
        self.bn1 = nn.BatchNorm2d(planes[0]) # 标准化
        self.relu1 = nn.LeakyReLU(0.1) # 激活函数
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False) # 3*3卷积64通道 扩张通道数
        self.bn2 = nn.BatchNorm2d(planes[1]) # 标准化
        self.relu2 = nn.LeakyReLU(0.1) # 激活函数

    def forward(self, x):
        # 残差块
        residual = x

        # 两组卷积+标准化+激活函数
        out = self.conv1(x) # 第一组
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out) # 第二组
        out = self.bn2(out)
        out = self.relu2(out)

        #将输出和残差边相加，这样就完成了一个前向传播的残差
        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        '''
        网络的初始化
        '''
        self.inplanes = 32 # 卷积的通道数
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) # 二维卷积
        self.bn1 = nn.BatchNorm2d(self.inplanes) # nn.BatchNorm2d()函数是进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.relu1 = nn.LeakyReLU(0.1) # 表示使用LeakyReLU激活函数,后面的参数表示x<0时激活函数的斜率。这个激活函数能解决深层神经网络梯度消失的问题。

        self.layer1 = self._make_layer([32, 64], layers[0]) # 这里的_make_layer代表残差块 layer1到layer5分别是五层残差块 第一个参数中的两个值分别是1*1和3*3卷积核的通道数
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        '''
        残差块
        planes：通道数

        '''
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False))) # 卷积
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1]))) # 标准化
        layers.append(("ds_relu", nn.LeakyReLU(0.1))) # 激活函数
        # 加入darknet模块   
        self.inplanes = planes[1]
        for i in range(0, blocks): # block规定了堆叠残差块的循环次数，对应101行model = DarkNet([1, 2, 8, 8, 4])中的参数。
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x) # 88 89 90这三行是主干网络中第一个卷积块的卷积、标准化和激活函数
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x) # 第一个残差块
        x = self.layer2(x) # 第二个残差块
        out3 = self.layer3(x) # 特征金字塔的一个输出 52*52*256
        out4 = self.layer4(out3) # 特征金字塔的一个输出 26*26*512
        out5 = self.layer5(out4) # 特征金字塔的一个输出 13*13*1024

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4]) # 这里的1，2，8，8，4对应的是主干网络中残差块的使用次数
    if pretrained: # 载入预训练
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
