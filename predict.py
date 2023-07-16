# ----------------------------------------------------#
#   单张图片预测，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import numpy as np
from PIL import Image
from nets.unet_predicting import Predict
from utils.utils import plot_matrix

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    predict = Predict()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    cm = True
    name_classes = ["Background", "Fixed", "Jitter", "Stagger", "Sliding", "D&S", "Wobulated", "noise"]
    # -------------------------------------------------------------------------#
    #   val                 指定了是否进行预测的准确度计算
    #   dir_val_path        指定了用于检测预测精准度的标签的文件夹路径
    # -------------------------------------------------------------------------#
    val = True
    dir_val_path = "img_val/"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img_in/"
    dir_save_path = "img_out/"
    # if mode == "predict":
    #     '''
    #     predict.py有几个注意点
    #     1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
    #     具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
    #     2、如果想要保存，利用r_image.save("img.jpg")即可保存。
    #     3、如果想要原图和分割图不混合，可以把blend参数设置成False。
    #     4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
    #     seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
    #     for c in range(self.num_classes):
    #         seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    #         seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    #         seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
    #     '''
    #     while True:
    #         img = input('Input image filename:')
    #         img_path = os.path.join('img_in', img)
    #         try:
    #             image = Image.open(img_path).convert('L')
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             r_image = predict.pred_image(image, count=count, name_classes=name_classes, val=val, val_path=val_path)
    #             r_image.save(os.path.join(dir_save_path, img))
    #             print("predict done...")
    if mode == "dir_predict":
        import os
        from tqdm import tqdm

        total_pred_acc = []
        total_hist = np.empty((8, 8))
        file_names = os.listdir(dir_origin_path)
        file_nums = len(file_names)
        for file_name in tqdm(file_names):
            if file_name.lower().endswith('.png'):
                image_path = os.path.join(dir_origin_path, file_name)
                val_path = os.path.join(dir_val_path, file_name)
                image = Image.open(image_path)
                r_image, pred_acc, hist = predict.pred_image(image, val=val, cm=cm,
                                                             val_path=val_path)
                if val:
                    total_pred_acc.append(pred_acc)
                if cm:
                    total_hist += hist
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, file_name))
            elif file_name.lower().endswith('.npy'):
                npy_path = os.path.join(dir_origin_path, file_name)
                npyfile = np.load(npy_path)
                file_name = str(file_name.split('.')[0]) + '.png'
                val_path = os.path.join(dir_val_path, file_name)
                r_image, pred_acc = predict.pred_npyfile(npyfile, val=val, val_path=val_path)
                if val:
                    total_pred_acc.append(pred_acc)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, file_name))
            else:
                raise AssertionError("Please specify the correct form: 'PNG' or 'NPY'.")
        if val:
            print('total_pred_acc: ', round((np.array(total_pred_acc).sum()) / file_nums, 3))
        if cm:
            plot_matrix(total_hist, [0, 1, 2, 3, 4, 5, 6, 7], title='Confusion matrix of RDNet(%)', name='1.png',
                        axis_labels=name_classes)

        print("predict done...")
    else:
        raise AssertionError("Please specify the correct mode: 'predict' or 'dir_predict'.")
