#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo3 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.config import Config
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
class YOLO(object):
    #
    _defaults = {
        "model_path": 'logs\Epoch1-Total_Loss63.1416-Val_Loss15.9550.pth',
        "classes_path": 'model_data/voc_classes.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self): # 载入目标包含的类数
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = YoloBody(self.config)

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 利用GPU进行计算
        state_dict = torch.load(self.model_path, map_location=device) # 载入权重文件
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # 图片处理
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1]))) # 对图片RESIZE并加灰条
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0 # 归一化
        photo = np.transpose(photo, (2, 0, 1)) # 在pytorch中通道数在第一个，所以在这调整顺序
        photo = photo.astype(np.float32) # 转换数据类型
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images) # 将numpy转换成tenor类型
        if self.cuda:
            images = images.cuda()
        # 放入网络中进行预测并画框
        with torch.no_grad():
            outputs = self.net(images) # 图片放入网络中
            output_list = []
            for i in range(3): # 特征层解码，因为特征金字塔有三个尺度的输出，所以要循环三次，将三个特征层全部解码。
                output_list.append(self.yolo_decodes[i](outputs[i])) # 解码：调整先验框
            output = torch.cat(output_list, 1) # 将预测结果堆叠起来
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=0.3) # non_max_suppression()是进行非极大抑制
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence # 将框框的置信度和类的置信度相乘进行判断
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5] # 下面这三行是将置信度较高的筛选出来
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        '''
        目前框框的位置是相对于有灰条图片左上角的位置。去掉灰条要转换为原图的左上角的位置。
        yolo_correct_boxes函数就是完成这样的坐标变换
        '''
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')) # 定义字体

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0] # 定义框框的宽度
        # 下面的代码就是用来画图的
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c] # 获得类的名称
            score = top_conf[i] # 获得得分
            # 获得位置信息
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font) # 在框上写字
            del draw
        return image

