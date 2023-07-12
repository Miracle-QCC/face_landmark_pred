import copy
import os, cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import random
import time
from scipy.integrate import simps
import torch.nn.functional as F




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class FocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.6, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()


def focalloss(inputs, targets, alpha = 0.6, gamma=2, flag_num=2):
    alpha = torch.tensor([alpha, 1 - alpha]).to(device)
    new_targets = torch.where(targets == 1, 1.0, 0.0).to(device)
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, new_targets, reduction='none')
    new_targets = new_targets.type(torch.long)
    at = alpha.gather(0, new_targets.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss * (targets != flag_num)
    return F_loss.mean()

def focalloss_blur(inputs, targets, alpha = 0.6, gamma=2):
    """
    针对blurness检测，label=2时表示模糊处理，其余0，1都不是
    :param inputs:
    :param targets:
    :param alpha:
    :param gamma:
    :return:
    """
    new_targets = torch.where(targets==2 ,1.0, 0.0).to(device)
    alpha = torch.tensor([alpha, 1 - alpha]).to(device)
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, new_targets, reduction='none')
    new_targets = new_targets.type(torch.long)
    at = alpha.gather(0, new_targets.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()

def get_label(data_name, label_file, task_type=None):
    label_path = os.path.join('data', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new

def get_label_5(data_name, label_file, task_type=None):
    label_path = os.path.join(data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new

def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
        
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def compute_loss_pip(outputs_map, outputs_local_x, outputs_local_y, outputs_nb_x, outputs_nb_y, labels_map, labels_local_x, labels_local_y, labels_nb_x, labels_nb_y,  criterion_cls, criterion_reg, num_nb):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map, labels_map)
    loss_x = criterion_reg(outputs_local_x_select, labels_local_x_select)
    loss_y = criterion_reg(outputs_local_y_select, labels_local_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)
    return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y


def compute_loss_pip_5(outputs_map, outputs_local_x, outputs_local_y, labels_map, labels_local_x, labels_local_y,  criterion_cls, criterion_reg, label, score):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)

    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)

    flag = label.repeat_interleave(5, dim=0)
    outputs_local_x_select_ = outputs_local_x_select * flag
    outputs_local_y_select_ = outputs_local_y_select * flag
    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map, labels_map)
    loss_x = criterion_reg(outputs_local_x_select_, labels_local_x_select)
    loss_y = criterion_reg(outputs_local_y_select_, labels_local_y_select)

    score = F.sigmoid(score)
    loss_score = focalloss(score, label)

    return loss_map, loss_x, loss_y, loss_score

def compute_loss_pip_5_(x_pred, y_pred, score, x_label, y_label, label, criterion_reg):

    x_pred = x_pred.squeeze()
    y_pred = y_pred.squeeze()

    x_pred = x_pred * label.repeat(1,5)
    y_pred = y_pred * label.repeat(1,5)

    loss_x = criterion_reg(x_pred, x_label)
    loss_y = criterion_reg(y_pred, y_label)

    loss_x = torch.sqrt(loss_x)
    loss_y = torch.sqrt(loss_y)

    score = F.sigmoid(score)
    score = score.view(-1,1)
    loss_score = focalloss(score, label)

    return loss_x, loss_y, loss_score


def compute_loss_pip_5_blurness(x_pred, y_pred, score, blur_score, x_label, y_label, label, criterion_reg):
    """
    增加了一维预测blurness
    :param x_pred:
    :param y_pred:
    :param score:
    :param x_label:
    :param y_label:
    :param label: 0,1,2
    :param criterion_reg:
    :return:
    """
    x_pred = x_pred.squeeze()
    y_pred = y_pred.squeeze()

    x_pred = x_pred * (label == 1)
    y_pred = y_pred * (label == 1)

    loss_x = criterion_reg(x_pred, x_label)
    loss_y = criterion_reg(y_pred, y_label)

    loss_x = torch.sqrt(loss_x)
    loss_y = torch.sqrt(loss_y)

    score = F.sigmoid(score)
    score = score.view(-1, 1)
    loss_score = focalloss(score, label)

    # loss blur
    blur_score = F.sigmoid(blur_score)
    blur_score = blur_score.view(-1,1)
    loss_blur = focalloss_blur(blur_score, label)
    return loss_x, loss_y, loss_score, loss_blur

def train_model(det_head, net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            if det_head == 'pip':
                inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y = data
                inputs = inputs.to(device)
                labels_map = labels_map.to(device)
                labels_x = labels_x.to(device)
                labels_y = labels_y.to(device)
                labels_nb_x = labels_nb_x.to(device)
                labels_nb_y = labels_nb_y.to(device)
                outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
                loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
                loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y + reg_loss_weight*loss_nb_x + reg_loss_weight*loss_nb_y
            else:
                print('No such head:', det_head)
                exit(0)
            # TODO  测试L1 loss
            loss = 0 * loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if det_head == 'pip':
                    print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                    logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                else:
                    print('No such head:', det_head)
                    exit(0)
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if epoch%(save_interval-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            print(filename, 'saved')
        scheduler.step()
    return net

def train_model_5(det_head, net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, optimizer, num_epochs, scheduler, save_dir, save_interval, device, logger):
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        logger.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):

            if det_head == 'pip':
                img_names, inputs, target, label = data
                inputs = inputs.to(device)
                target = target.to(device)
                x_label = target[:,::2]
                y_label = target[:,1::2]

                # img = inputs[0]
                # img = img.permute(1, 2, 0)
                # img = img * 128 + 128
                # img = img.cpu().numpy()
                # img = img.astype(np.uint8)
                #
                # # im_ = np.array(img)
                # b, g, r = cv2.split(img)
                # img = cv2.merge([r, g, b])
                # for i in range(5):
                #     cv2.circle(img, (int(x_label[0, i] * 128), int(y_label[0, i] * 128)), 1, (0, 0, 255), 2)
                #     cv2.putText(img, str(i), (int(x_label[0, i] * 128), int(y_label[0, i] * 128)), 1, 1,
                #                 (0, 0, 255), 2)
                #
                # cv2.imshow("x", img)
                # cv2.waitKey(0)

                label = label.to(device).float()
                x_pred, y_pred, score = net(inputs)
                loss_x, loss_y, loss_score = compute_loss_pip_5_(x_pred, y_pred, score, x_label, y_label, label, criterion_reg)
                loss = reg_loss_weight*loss_x + reg_loss_weight*loss_y + loss_score * cls_loss_weight
            else:
                logger.info('No such head:', det_head)
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if det_head == 'pip':
                    logger.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <score loss: {:.6f}> '.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_score.item()))
                    # logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> '.format(
                    #     epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_score.item()))
                else:
                    logger.info('No such head:' + det_head)
                    exit(0)
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if epoch%(11-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            logger.info(filename + 'saved')
        scheduler.step()
    return net


def train_model_5_blurness(det_head, net, train_loader, criterion_reg, cls_loss_weight, reg_loss_weight, optimizer, num_epochs, scheduler, save_dir, device, logger):
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        logger.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):

            if det_head == 'pip':
                img_names, inputs, target, label = data
                inputs = inputs.to(device)
                target = target.to(device)
                x_label = target[:,::2]
                y_label = target[:,1::2]


                label = label.to(device).float()
                x_pred, y_pred, score, blur_score = net(inputs)
                loss_x, loss_y, loss_score, loss_blur = compute_loss_pip_5_blurness(x_pred, y_pred, score, blur_score, x_label, y_label, label, criterion_reg)
                loss = reg_loss_weight*loss_x + reg_loss_weight*loss_y + loss_score * cls_loss_weight + loss_blur * reg_loss_weight
            else:
                logger.info('No such head:', det_head)
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if det_head == 'pip':
                    logger.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <score loss: {:.6f}> <blur loss: {:.6f}> '.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_score.item(), loss_blur.item() * cls_loss_weight))
                    # logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> '.format(
                    #     epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_score.item()))
                else:
                    logger.info('No such head:' + det_head)
                    exit(0)
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if epoch%(11-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            logger.info(filename + 'saved')
        scheduler.step()
    return net


def forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb):
    net.eval()
    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        assert tmp_batch == 1

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
        max_ids = torch.argmax(outputs_cls, 1)
        max_cls = torch.max(outputs_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)
        outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * input_size / net_stride
        tmp_y /= 1.0 * input_size / net_stride

        tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
        tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, num_nb)
        tmp_nb_y = tmp_nb_y.view(-1, num_nb)
        tmp_nb_x /= 1.0 * input_size / net_stride
        tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc
