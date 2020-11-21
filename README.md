# yolo3-pytorch
#### 基于Pytorch的YOLO v3目标检测代码



所需环境： torch == 1.2.0

# 训练步骤
1. 数据集使用VOC格式。将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中
2. 在训练前利用voc2yolo4.py文件生成对应的txt。
3. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。注意不要使用中文标签，文件夹中不要有空格！ 
'''
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
'''
4. 此时会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。
5. 在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件，示例如下：   
'''
classes_path = 'model_data/new_classes.txt'    
model_data/new_classes.txt文件内容为：

cat
dog
...
'''
6. 修改utils/config.py里面的classes，使其为要检测的类的个数。
7. 运行train.py即可开始训练。

# 训练时注意:
utils\config.py中的classes的数量需要修改,若使用聚类算法，先验框的尺寸也在此调整。
train.py中训练集和验证集的划分，关系到后面计算MAP。

# 使用自己的权重进行图片预测或摄像头预测：
1. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
'''
_defaults = {
    "model_path": 'model_data/yolo_weights.pth',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt,
    "score" : 0.5,
    "iou" : 0.3,
    # 显存比较小可以使用416x416
    # 显存比较大可以使用608x608
    "model_image_size" : (416, 416)
}
'''
2. 如进行图片预测：运行predict.py后，输入图片地址即可。
3. 如进行视频预测：运行video.py。
# 计算MAP：
1. 运行get_dr_txt.py，此时会在input\detection-results中生成预测结果的txt文件。每个txt对应一个图片
2. 运行get_gt_txt.py，此时会在input\ground-truth中生成真实标签的txt文件。每个txt对应一个图片
3. 运行get_map.py，此时会在results中生成所需的结果。

基础知识： 卷积后尺寸计算： 输入大小为ww 卷积核大小ff 步长s 填充像素数p 卷积后的尺寸为 n=(w-f+2p)/s+1

YOLOV3笔记： 
1 首先将图像调整到416416的大小。为了防止图像失真（长宽比不是1：1的话），会将空白部分用灰色填充。 
2 将图像分别分成1313、2626、5252的网格。不同尺度的网格用来检测不同尺寸的物体。
3 每个网格点负责右下角区域的预测，只要物体中心点落在这个区域里，这个物体就由这个网格来确定。

# YOLOV3实现过程 ！！！注意：最后一个数是通道数，但在实际的代码中，通道数在batch_size后面的一个。 
1 主干特征提取网络DarkNet-53 
  ship 1: iuput (batch_size, 416, 416, 3) 
  ship 2: conv2D 3233 (batch_size, 208, 80, 64) 
  ship 3: Residual Block 164 (batch_size, 208, 280, 64) 
  ship 4: Residual Block 2128 (batch_size, 104, 104, 128) 
  ship 5: Residual Block 8256 (batch_size, 52, 52, 256) ->concat 
  ship 6: Residual Block 8512 (batch_size, 26, 26, 512) ->concat 
  ship 7: Residual Block 41024 (batch_size, 13, 13, 1024) ->concat 
2 特征金字塔 对ship 7的输出的特征图（13， 13，1024）进行五次卷积，结果记为out 1。这个结果有两个走向。 走向1： 对out 1进行分类和回归预测，实际上是两次卷积，一次33的卷积，一次11的卷积。最后得到（13，13，75）->(13, 13 ,253) ->(13, 13, 3*(20+1+4))。3代表三个先验框，20代表20个类别的置信度，1代表是否有物体，4代表预测框的坐标。 走向2： 上采样，与ship 6的结果进行连接。连接的结果记为out 2。这个结果有两个走向。 对out 2进行预测（同上面走向1）得到（26，26，75） out 2再进行上采样，与ship 5的结果连接，再进行预测，得到（52, 52, 75）

# 主要部分代码的结构：
darknet.py：定义了主干特征提取网络DarkNet-53的结构,最后将主干网络存储在变量model中。
yolo3.py：定义了特征金字塔部分的结构，将最后预测的结果保存out0 out1 out2中。out0是最大尺度的结果，out2是最小尺度的结果。
utils/utils.py：解码，对先验框进行调整。
utils\config.py：原始先验框的设定，跟nets\yolo_training.py密切相关
predict.py: 对单张图片进行预测
nets\yolo_training.py：定义训练阶段的结构
train.py：启动训练，定义训练过程
yolo.py：定义预测阶段的结构

iou部分:utils\utils.py
LOSS部分：train.py
lr和Batch_size部分：train.py
