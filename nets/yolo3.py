'''
特征金字塔部分
'''
import torch
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    '''
    定义一个卷积块，包括一次卷积，一次标准化和一个激活函数
    '''
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    '''
    最后的那七次卷积
    '''
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1), # 1*1卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3), # 3*3卷积提取特征
        conv2d(filters_list[1], filters_list[0], 1), # 1*1卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3), # 3*3卷积提取特征
        conv2d(filters_list[1], filters_list[0], 1), # 1*1卷积调整通道数
        conv2d(filters_list[0], filters_list[1], 3), # 下面这两个卷积是分类预测和回归预测
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53(None) # 将darknet.py中获得的主干网络的结构保存在.backbone属性中。

        out_filters = self.backbone.layers_out_filters
        #  last_layer0 3*（5+num_classes)=3*(5+20)=3*(4+1+20)=75 这部分是处理out5的特征层
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"]) # final_out_filter0就是75 是特征图的参数
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0) # make_last_layers是七次卷积，最后两次卷积是回归预测和分类预测

        #  embedding1 75 这部分是处理out4的特征层
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1) # 用1*1的卷积调整通道数
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 第一次上采样
        # 此处已经获得26,26,256的特征层
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1) # 在前向传播时两个尺度的特征层进行了堆叠。make_last_layers是七次卷积，最后两次卷积是回归预测和分类预测

        #  embedding2 75 这部分是处理out3的特征层
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1) # 1*1卷积调整通道数
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 第二次上采样
        # 此处已经获得52,52,128的特征层
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2) # 在前向传播时两个尺度的特征层进行了堆叠。make_last_layers是七次卷积，最后两次卷积是回归预测和分类预测


    def forward(self, x):
        def _branch(last_layer, layer_in): # 因为特征金字塔那七次卷积是在一起的，但结果要拆开保存，_branch就是分别存放这些结果
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in # 将特征金字塔部分那五次卷积的结果保存在out_branch中
            return layer_in, out_branch # 将特征金字塔部分回归预测和分类预测的结果保存在layer_in
        #  backbone
        x2, x1, x0 = self.backbone(x) # 获取主干特征提取网络
        '''
        x2: Out3对应的特征层
        x1: Out4对应的特征层
        x0: Out5对应的特征层
        '''
        #  yolo branch 0
        out0, out0_branch = _branch(self.last_layer0, x0) # 将五次卷积和最后两次卷积结果分开保存，out0是最后两次的结果，out0_branch是前五次的结果

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch) # 1*1卷积调整通道数
        x1_in = self.last_layer1_upsample(x1_in) # 上采样
        x1_in = torch.cat([x1_in, x1], 1) # 不同尺度的特征层进行堆叠
        out1, out1_branch = _branch(self.last_layer1, x1_in) # 将五次卷积和最后两次卷积结果分开保存，out1是最后两次的结果，out1_branch是前五次的结果

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch) # 1*1卷积调整通道数
        x2_in = self.last_layer2_upsample(x2_in) # 上采样
        x2_in = torch.cat([x2_in, x2], 1) # 不同尺度的特征层进行堆叠
        out2, _ = _branch(self.last_layer2, x2_in) # 将五次卷积和最后两次卷积结果分开保存，out2是最后两次的结果.
        return out0, out1, out2

