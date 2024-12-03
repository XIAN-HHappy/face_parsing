#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *



class FaceMask(Dataset):
    def __init__(self, rootpth,img_size, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth
        self.img_size = img_size

        self.imgs = os.listdir(os.path.join(self.rootpth, 'CelebA-HQ-img'))

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(osp.join(self.rootpth, 'CelebA-HQ-img', impth))
        img = img.resize((self.img_size,self.img_size), Image.BILINEAR)
        label = Image.open(osp.join(self.rootpth, 'mask_{}'.format(self.img_size), impth[:-3]+'png')).convert('P')
        # print(np.unique(np.array(label)))
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return len(self.imgs)
