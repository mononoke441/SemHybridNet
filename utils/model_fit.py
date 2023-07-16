import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, f_score, classify_accuracy
from tqdm import tqdm

from utils.utils import get_lr

'''
def fit_one_epoch(model_train, model,loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, save_period,
                  save_dir, dataset_form, preprocess_input, local_rank=0):
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        # print('imgs.shape: ',imgs.shape)
        # print('pngs.shape: ', pngs.shape)
        # print('labels.shape: ', labels.shape)
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model_train(imgs)
        # print('outputs.shape: ', outputs.shape)
        # ----------------------#
        #   损失计算
        # ----------------------#
        if focal_loss:
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
        with torch.no_grad():
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            _f_score = f_score(outputs, labels)
            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train, dataset_form, preprocess_input)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
'''


def fit_one_epoch(model_train, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, save_period,
                  save_dir, dataset_form, preprocess_input, local_rank=0, using_gpu=0, accumulation_steps=2):
    """
    :param model_train:
    :param loss_history:
    :param eval_callback:
    :param optimizer:
    :param epoch:
    :param epoch_step:
    :param epoch_step_val:
    :param gen:
    :param gen_val:
    :param Epoch:
    :param cuda:
    :param dice_loss:
    :param focal_loss:
    :param cls_weights:
    :param num_classes:
    :param save_period:
    :param save_dir:
    :param dataset_form:
    :param preprocess_input:
    :param local_rank:
    :return:
    """
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0
    # val_acc = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    # 传进来的参数已没有.train
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        # print('imgs.shape: ',imgs.shape)
        # print('pngs.shape: ', pngs.shape)
        # print('labels.shape: ', labels.shape)
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(using_gpu)
                pngs = pngs.cuda(using_gpu)
                labels = labels.cuda(using_gpu)
                weights = weights.cuda(using_gpu)

        # optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model_train(imgs)
        # print('outputs.shape: ', outputs.shape)
        # ----------------------#
        #   损失计算
        # ----------------------#
        if focal_loss:
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
        # mini batch size
        loss = loss / accumulation_steps
        with torch.no_grad():
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)
        loss.backward()
        if (iteration + 1) % accumulation_steps == 0:
            optimizer.zero_grad()
            optimizer.step()
        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(using_gpu)
                pngs = pngs.cuda(using_gpu)
                labels = labels.cuda(using_gpu)
                weights = weights.cuda(using_gpu)
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # _val_acc = classify_accuracy(outputs, pngs)
            _f_score = f_score(outputs, labels)
            val_loss += loss.item()
            val_f_score += _f_score.item()
            # val_acc += _val_acc

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train, dataset_form, preprocess_input)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model_train.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model_train.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save({'model': model_train.state_dict(),
                    'epoch': epoch + 1},
                   os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # ----------------------#
        # 前向传播
        # ----------------------#
        outputs = model_train(imgs)
        # ----------------------#
        #   损失计算
        # ----------------------#
        if focal_loss:
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice

        with torch.no_grad():
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model_train.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model_train.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model_train.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
