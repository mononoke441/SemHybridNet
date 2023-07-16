import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.model_fit import fit_one_epoch

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    using_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Cuda = True
    # -----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # -----------------------------------------------------#
    num_classes = 8
    # -----------------------------------------------------#
    #   input_shape     输入图片的大小，32的倍数
    # -----------------------------------------------------#
    input_shape = [512, 512]
    # -----------------------------------------------------#
    #   preprocess_input     加载数据集时是否进行数据预处理
    # -----------------------------------------------------#
    preprocess_input = False
    # -----------------------------------------------------#
    #   model_name      使用的模型
    #                   unet
    #                   attention_unet
    #                   seg_net
    #                   SegNext
    #                   transunet
    #                   TransAttUnet
    #                   SETR
    #                   SemHybridNet
    # -----------------------------------------------------#
    model_name = 'SemHybridNet'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # -----------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # -----------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Epoch = 150
    batch_size = 4
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 30
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5
    # ---------------------------------------------------------#
    radar_path = ''
    # ------------------------------#
    #   表示数据集的格式，PNG或NPY
    #   默认情况下的格式为PNG
    # ----------------------------- #
    dataset_form = 'PNG'
    # ------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ------------------------------------------------------------------#
    dice_loss = True
    # ------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ------------------------------------------------------------------#
    focal_loss = False
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#
    num_workers = 4
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    local_rank = 0
    if torch.cuda.is_available():
        # ngpus_per_node = torch.cuda.device_count()
        device = torch.device('cuda', using_gpu)
    else:
        device = torch.device('cpu')
    # ----------------------#
    #   加载使用的模型
    # ----------------------#
    if model_name == 'unet':
        from nets.unet import Unet

        model = Unet(in_channel=1, out_channel=64, classes_num=num_classes).train()
    if model_name == 'attention_unet':
        from nets.attention_unet import AttentionUnet

        model = AttentionUnet(num_classes=num_classes, preprocess_flag=preprocess_input).train()
    if model_name == 'seg_net':
        from nets.seg_net import SegNet

        # 使用pretrained vgg_16作为backbone
        # pretrained = True
        # freeze_bn
        # if batch_size < 4:  # batch_size过小要冻结BN层
        # freeze_bn = True
        # else:
        # freeze_bn = False
        model = SegNet(in_channel=1, num_classes=num_classes, preprocess_flag=preprocess_input,
                       pretrained=pretrained).train()
    if model_name == 'SegNext':
        from nets.SegNext import SegNext

        model = SegNext(num_classes=num_classes, preprocess_flag=preprocess_input, in_channel=1).train()
    if model_name == 'transunet':
        from nets.transunet import TransUNet

        model = TransUNet(img_dim=input_shape[0],
                          in_channels=1,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=num_classes).train()
    if model_name == 'TransAttUnet':
        # 没有preprocess
        from nets.TransAttUnet import UNet_Attention_Transformer_Multiscale as TransAttUnet

        model = TransAttUnet(1, num_classes).train()
    if model_name == 'SETR':
        from nets.SETR import SETRModel as SETR

        model = SETR(patch_size=(32, 32),
                     in_channels=1,
                     out_channels=num_classes,
                     hidden_size=1024,
                     sample_rate=4,
                     num_hidden_layers=8,
                     num_attention_heads=16,
                     decode_features=[512, 256, 128, 64]).train()
    if model_name == 'SemHybridNet':
        from nets.SemHybridNet import SemHybridNet

        model = SemHybridNet(img_dim=512,
                      in_channel=1,
                      out_channel=128,
                      num_classes=num_classes,
                      head_num=8,
                      mlp_dim=512,
                      block_num=1,
                      patch_dim=16,
                      preprocess_flag=None)
    # ----------------------#
    #   初始化模型参数
    # ----------------------#
    if not pretrained:
        weights_init(model)
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    model_train = model
    if Cuda:
        # model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(radar_path, "DataSets/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(radar_path, "DataSets/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    print('训练数据个数：', num_train)
    num_val = len(val_lines)
    print('验证数据个数：', num_val)
    # ------------------------------------------------------#
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        # -------------------------------------------------------------------#
        #   直接设置batch_size为batch_size
        # -------------------------------------------------------------------#
        batch_size = batch_size
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]
        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, preprocess_input, radar_path, dataset_form)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, preprocess_input, radar_path, dataset_form)
        train_sampler = None
        val_sampler = None
        shuffle = True
        '''
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)
        '''

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, radar_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        # ------------------------------------#
        #   检查是否有之前训练一半的模型，继续训练
        # ------------------------------------#
        if pretrained:
            from pathlib import Path

            p = Path(__file__)
            ckpt_name = 'last_epoch_weights.pth'
            ckpt_path = p.parent.parent.joinpath('logs/', ckpt_name)
            if os.path.exists(ckpt_path):
                print('Restart from checkpoint {}'.format(ckpt_path))
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                model_train.load_state_dict(checkpoint['model'])
                model_train.cuda()
                Init_Epoch = checkpoint['epoch']
            else:
                pass

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, Epoch):
            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 16
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, save_period, save_dir, dataset_form, preprocess_input,
                          local_rank=local_rank,
                          using_gpu=using_gpu)
        if local_rank == 0:
            loss_history.writer.close()
