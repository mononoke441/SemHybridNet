import os

import matplotlib
import torch
import torch.nn.functional as F

from nets.unet_training import compute_mIoU
from utils.utils import preprocess_input,plot_matrix

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 1, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.accs = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
            with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image, dataset_form, flag):
        if dataset_form == 'PNG':
            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]
            image_data = np.reshape(image, image.size + (1,))
        elif dataset_form == 'NPY':
            if len(image.shape) == 2:  # channel=1,转换通道H,W,C至C,H,W
                original_h = np.array(image).shape[0]
                original_w = np.array(image).shape[1]
                # image为H,W，添加channel维度
                image_data = np.reshape(image, image.shape + (1,))
            else:  # channel=2
                original_h = np.array(image).shape[1]
                original_w = np.array(image).shape[2]
                # image为C,H,W无需添加channel维度和转换通道
                image_data = image
        else:  # type error
            raise AssertionError("Please specify the correct form: 'PNG' or 'NPY'.")

        # 添加batch_size维度
        if flag:
            image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float64)), [2, 0, 1]), 0)
        else:
            image_data = np.expand_dims(np.transpose(np.array(image_data, np.float64), [2, 0, 1]), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测，取出除batch_size维度的数据？
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # ---------------------------------------------------#
            #   将图片大小规范
            #   cv2.resize用于大小不一
            #   pr = np.reshape(pr, (original_w, original_h))
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval, dataset_form, preprocess_flag):
        name_classes = ["Background", "Fixed", "Jitter", "Stagger", "Sliding", "D&S", "Wobulated", "noise"]
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                if dataset_form == 'PNG':
                    # -------------------------------#
                    #   从文件中读取图像
                    # -------------------------------#
                    image_path = os.path.join(self.dataset_path, "PNGImages/" + image_id + ".png")
                    image = Image.open(image_path).convert('L')
                elif dataset_form == 'NPY':
                    # -------------------------------#
                    #   从中加载numpy文件
                    # -------------------------------#
                    image_path = os.path.join(self.dataset_path, "PNGImages/" + image_id + ".npy")
                    image = np.load(image_path)
                else:
                    raise AssertionError("Please specify the correct form: 'PNG' or 'NPY'.")
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                image = self.get_miou_png(image, dataset_form, preprocess_flag)
                image.save(os.path.join(pred_dir, image_id + ".png"))

            print("Calculate miou.")
            hist, IoUs, _, _, Accuracy = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes,
                                                   None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100
            temp_acc = Accuracy * 100
            self.mious.append(temp_miou)
            self.accs.append(temp_acc)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
                f.write(str(temp_acc))
                f.write("\n")
            plot_matrix(hist, [0, 1, 2, 3, 4, 5, 6, 7], title='Confusion matrix of RDNet(%)', name=str(epoch)+'.png',
                        axis_labels=name_classes)
            # 绘制mIoU图像
            plt.figure()  # 创建自定义图像
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            # 绘制Accuracy图像
            plt.figure()
            plt.plot(self.epoches, self.accs, 'red', linewidth=2, label='val acc')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('An Accuracy Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
