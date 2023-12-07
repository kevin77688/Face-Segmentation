import os.path as osp
import os
import cv2
from transform import *
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

face_data = '/home/kevin/Code/CV_Workshop/Face_Competition/data/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/kevin/Code/CV_Workshop/Face_Competition/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/kevin/Code/CV_Workshop/Face_Competition/data/CelebAMask-HQ/CelebAMask-HQ-combined_mask'
counter = 0
total = 0

shutil.rmtree(mask_path)
os.makedirs(mask_path)

for i in tqdm(range(15), leave=True):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in tqdm(range(i * 2000, (i + 1) * 2000), leave=False):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        # print(j)

print(counter, total)