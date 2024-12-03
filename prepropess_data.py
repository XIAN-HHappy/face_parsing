# -*- encoding: utf-8 -*-
#function : 训练样本预处理

import os
import os.path as osp
import cv2
from transform import *
from PIL import Image

if __name__ == "__main__":

    image_size = 512# 样本分辨率

    face_data = './CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = './CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = './CelebAMask-HQ/mask_{}'.format(image_size)

    if not os.path.exists(mask_path):
        os.mkdir(mask_path)

    counter = 0
    total = 0
    for i in range(15):

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i * 2000, (i + 1) * 2000):

            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))

                    mask[sep_mask == 225] = l
            if image_size != 512:
                mask = cv2.resize(mask,(image_size,image_size),interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            print(j)

    print(counter, total)
