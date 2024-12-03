# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
import torch.optim as Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os.path as osp
import time
import datetime
import argparse

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(fintune_model,image_size,lr0,path_data,model_exp):

    # config 训练配置
    max_epoch = 1000
    n_classes = 19
    n_img_per_gpu = 16
    n_workers = 12
    cropsize = [int(image_size*0.8),int(image_size*0.8)]

    # DataLoader 数据迭代器
    ds = FaceMask(path_data,img_size = image_size, cropsize=cropsize, mode='train')

    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100
    # 构建模型
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    net = BiSeNet(n_classes=n_classes)
    net = net.to(device)
    # 加载预训练模型
    if os.access(fintune_model,os.F_OK) and (fintune_model is not None):# checkpoint
        chkpt = torch.load(fintune_model, map_location=device)
        net.load_state_dict(chkpt)
        print('load fintune model : {}'.format(fintune_model))
    else:
        print('no fintune model')
    # 构建损失函数
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = lr0
    # 构建优化器
    optim = Optimizer.SGD(
            net.parameters(),
            lr = lr_start,
            momentum = momentum,
            weight_decay = weight_decay)

    # train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    # diter = iter(dl)
    epoch = 0
    flag_change_lr_cnt = 0 # 学习率更新计数器
    init_lr = lr_start # 学习率

    best_loss = np.inf
    loss_mean = 0. # 损失均值
    loss_idx = 0. # 损失计算计数器
    # 训练
    print('start training ~')
    it = 0
    for epoch in range(max_epoch):
        net.train()
        # 学习率更新策略
        if loss_mean!=0.:
            if best_loss > (loss_mean/loss_idx):
                flag_change_lr_cnt = 0
                best_loss = (loss_mean/loss_idx)
            else:
                flag_change_lr_cnt += 1

                if flag_change_lr_cnt > 30:
                    init_lr = init_lr*0.1
                    set_learning_rate(optim, init_lr)
                    flag_change_lr_cnt = 0

        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器

        for i, (im, lb) in enumerate(dl):

            im = im.cuda()
            lb = lb.cuda()
            H, W = im.size()[2:]
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            out, out16, out32 = net(im)
            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)
            loss = lossp + loss2 + loss3

            loss_mean += loss.item()
            loss_idx += 1.

            loss.backward()
            optim.step()

            if it % msg_iter == 0:

                print('epoch <{}/{}> -->> <{}/{}> -> iter {} : loss {:.5f}, loss_mean :{:.5f}, best_loss :{:.5f},lr :{:.6f},batch_size : {},img_size :{}'.\
                format(epoch,max_epoch,i,int(ds.__len__()/n_img_per_gpu),it,loss.item(),loss_mean/loss_idx,best_loss,init_lr,n_img_per_gpu,image_size))

                if (it) % 500 == 0:
                    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                    torch.save(state, model_exp+'fp_{}_latest.pth'.format(image_size))
            it += 1
        torch.save(state, model_exp+'fp_{}_epoch-{}.pth'.format(image_size,epoch))

if __name__ == "__main__":
    image_size = 512
    lr0 = 1e-4
    model_exp = './model_exp/'
    path_data = './CelebAMask-HQ/'
    if not osp.exists(model_exp):
        os.makedirs(model_exp)


    loc_time = time.localtime()
    model_exp += time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    if not osp.exists(model_exp):
        os.makedirs(model_exp)

    fintune_model = './weights/fp0.pth'

    train(fintune_model,image_size,lr0,path_data,model_exp)
