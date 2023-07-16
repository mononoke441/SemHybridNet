import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.attention_unet import AttentionUnet
from nets.unet import Unet
from nets.SemHybridNet import SemHybridNet
from nets.seg_net import SegNet
from nets.SETR import SETRModel
from nets.transunet import TransUNet
from nets.SegNext import SegNext
from utils.utils import preprocess_input
from utils.utils import fast_hist


class Predict(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        # -------------------------------------------------------------------#
        "model_path": 'model/best_epoch_weights.pth',
        # -------------------------------------------------------------------#
        #   model_name是使用的模型
        # -------------------------------------------------------------------#
        "model_name": 'SemHybridNet',
        # --------------------------------#
        #   所需要区分的类的个数+1
        # --------------------------------#
        "num_classes": 8,
        # --------------------------------#
        #   输入图片的大小
        # --------------------------------#
        "input_shape": [512, 512],
        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU设置成False
        # --------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        self.colors = [(0, 0, 0),  # 背景
                       (255, 165, 0),  # 类别 1，恒参，橙色
                       (255, 255, 0),  # 类别 2，抖动，黄色
                       (0, 255, 0),  # 类别 3，参差，绿色
                       (0, 0, 255),  # 类别 4，滑变，蓝色
                       (128, 0, 128),  # 类别 5，组变，紫色
                       (220, 20, 60),  # 类别 6，正弦，猩红色
                       (255, 255, 255)]  # 类别 7，噪音，白色
        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.get_unet()

    def get_unet(self):
        self.net = {
            'unet': Unet(in_channel=1, out_channel=64, classes_num=self.num_classes),
            'attention_unet': AttentionUnet(num_classes=self.num_classes),
            'RDNet': SemHybridNet(img_dim=512,
                           in_channel=1,
                           out_channel=128,
                           num_classes=self.num_classes,
                           head_num=8,
                           mlp_dim=512,
                           block_num=1,
                           patch_dim=16,
                           preprocess_flag=None)
        }[self.model_name]
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        else:
            device = torch.device('cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def pred_image(self, image, val=False, cm=False, val_path=None):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data = np.reshape(image, image.size + (1,))
        # -------------------------------------------------------#
        #   添加上batch_size维度
        # -------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # print("pr.shape", pr.shape)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        '''
        # if count:
        #     classes_nums = np.zeros([self.num_classes])
        #     total_points_num = orininal_h * orininal_w
        #     print('-' * 63)
        #     print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        #     print('-' * 63)
        #     for i in range(self.num_classes):
        #         num = np.sum(pr == i)
        #         ratio = num / total_points_num * 100
        #         if num > 0:
        #             print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
        #             print('-' * 63)
        #         classes_nums[i] = num
        #     print("classes_nums:", classes_nums)
        '''
        # -------------------------------------------------------#
        #   将新图片转换成Image的形式
        # -------------------------------------------------------#
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        pred_acc = -1
        hist = []
        if val:
            val_image = Image.open(val_path)
            val_image = np.array(val_image)
            pred_acc = round(((val_image == seg_img).sum()) / (orininal_h * orininal_w), 3)
            print('pred_acc: ', pred_acc)
        if cm:
            val_image = Image.open(val_path).convert('L')
            val_image = np.array(val_image)
            hist = fast_hist(val_image, seg_img, self.num_classes)
        image = Image.fromarray(np.uint8(seg_img))
        return image, pred_acc, hist

    def pred_npyfile(self, npyfile, val=True, val_path=None):
        if len(npyfile.shape) == 2:
            orininal_h = np.array(npyfile).shape[0]
            orininal_w = np.array(npyfile).shape[1]
            npyfile = np.reshape(npyfile, npyfile.shape + (1,))
            npy_data = np.transpose(np.array(npyfile, np.float32), (2, 0, 1))
        else:
            orininal_h = np.array(npyfile).shape[1]
            orininal_w = np.array(npyfile).shape[2]
            npy_data = npyfile
        # -------------------------------------------------------#
        #   添加上batch_size维度
        # -------------------------------------------------------#
        npy_data = np.expand_dims(preprocess_input(np.array(npy_data, np.float32)), 0)

        with torch.no_grad():
            images = torch.from_numpy(npy_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # print("pr.shape", pr.shape)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        # -------------------------------------------------------#
        #   将预测结果pr转换成Image的形式
        # -------------------------------------------------------#
        pred_acc = -1  # val=False时
        if val:
            val_image = Image.open(val_path)
            val_image = np.reshape(val_image, val_image.size)
            pred_acc = round(((val_image == pr).sum()) / (orininal_h * orininal_w), 3)
            print('pred_acc: ', pred_acc)
        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))
        return image, pred_acc
