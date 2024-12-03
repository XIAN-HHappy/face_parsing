# -*- encoding: utf-8 -*-

from model import BiSeNet
import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

def vis_parsing_maps(im, parsing_anno,x,y, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy()
    vis_parsing_anno_color = np.zeros((im.shape[0], im.shape[1], 3)) + 0

    face_mask = np.zeros((im.shape[0], im.shape[1]))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)# 获得对应分类的的像素坐标

        idx_y = (index[0]+y).astype(np.int)
        idx_x = (index[1]+x).astype(np.int)

        # continue
        vis_parsing_anno_color[idx_y,idx_x, :] = part_colors[pi]# 给对应的类别的掩码赋值

        face_mask[idx_y,idx_x] = 0.45
        # if pi in[1,2,3,4,5,6,7,8,10,11,12,13,14,17]:
        #     face_mask[idx_y,idx_x] = 0.35

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    face_mask = np.expand_dims(face_mask, 2)
    vis_im = vis_parsing_anno_color*face_mask + (1.-face_mask)*vis_im
    vis_im = vis_im.astype(np.uint8)

    return vis_im


def inference( img_size, image_path, model_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    print('model : {}'.format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        idx = 0
        for f_ in os.listdir(image_path):
            img_ = cv2.imread(image_path + f_)
            img = Image.fromarray(cv2.cvtColor(img_,cv2.COLOR_BGR2RGB))

            image = img.resize((img_size, img_size))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing_ = out.squeeze(0).cpu().numpy().argmax(0)
            idx += 1
            print('<{}> image : '.format(idx),np.unique(parsing_))

            parsing_ = cv2.resize(parsing_,(img_.shape[1],img_.shape[0]),interpolation=cv2.INTER_NEAREST)

            parsing_ = parsing_.astype(np.uint8)
            vis_im = vis_parsing_maps(img_, parsing_, 0,0,stride=1)

            # 保存输出结果
            test_result = './result/'
            if not osp.exists(test_result):
                os.makedirs(test_result)
            cv2.imwrite(test_result+"p_{}-".format(img_size)+f_,vis_im)

            cv2.namedWindow("vis_im",0)
            cv2.imshow("vis_im",vis_im)

            if cv2.waitKey(500) == 27:
                break
if __name__ == "__main__":
    img_size = 512 # 推理分辨率设置
    model_path = "./weights/fp_512.pth" # 模型路径
    image_path = "./images/"
    inference(img_size = img_size, image_path=image_path, model_path=model_path)
