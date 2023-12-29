import cv2
import numpy as np
import os
import math
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch
from math import dist
from scipy.spatial import distance


DATA_ROOT = '/home/kevin/Desktop/Save'
OUTPUT_DATA_ROOT = '/home/kevin/Desktop'
INPUT_IMAGE = '2098747557_1.jpg'
INPUT_INDEX = '0'
OUTPUT_IMAGE = 'alignment.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = [0.6, 0.6, 0.6]
mtcnn = MTCNN(
    thresholds=threshold,
    device=device
)


def detect_faces(img, toRGB=True):
    '''
        Input: original image
        Output: bounding box of faces
    '''
    # step = 0.1
    # if toRGB:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn = MTCNN(
        thresholds=threshold,
        device=device
    )
    try:
        faces, probs, landmarks = mtcnn.detect(img, landmarks=True)
        return faces
        h, w = img.shape[:2]
        center = (h // 2, w // 2)
        min_dist = max(h, w)
        index = 0
        for l_index in range(len(landmarks)):
            dist = distance.euclidean(landmarks[l_index][2], center)
            if dist < min_dist:
                min_dist = dist
                index = l_index
        # index = np.argmax(probs)
        face = faces[index]
        round_face_list = [round(pt) for pt in face]
        x_min, y_min, x_max, y_max = np.round(round_face_list)
        mask = np.zeros_like(img[:, :, 0], dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        cloned_image = img.copy()
        cloned_image[~mask] = (0, 0, 255)
        landmark = landmarks[index]
        print(tuple(landmark[2]))
        keypoints = {'left_eye': tuple(landmark[0]), 'right_eye': tuple(landmark[1]), 'nose': tuple(
            landmark[2]), 'mouth_left': tuple(landmark[3]), 'mouth_right': tuple(landmark[4])}
        return keypoints
    except:
        return None
        # max_index = threshold.index(max(threshold))
        # threshold[max_index] -= step
        # mtcnn = MTCNN(thresholds=threshold, device=device)


def remove_redundant(img, gray_img):
    faces = detect_faces(img)
    if faces is None:
        # print("No face detected\n\n\n\n\n\n")
        return gray_img
    h, w = img.shape[:2]
    img_center = (round(h // 2), round(w // 2))
    min_dist = max(h, w)
    index = 0
    for f_index in range(len(faces)):
        tl = (faces[f_index][0], faces[f_index][1])
        br = (faces[f_index][2], faces[f_index][3])
        box_center = (round((br[0] + tl[0]) // 2), round((br[1] + tl[1]) // 2))
        dist = distance.euclidean(box_center, img_center)
        if dist < min_dist:
            min_dist = dist
            index = f_index

    mask = np.zeros_like(gray_img, dtype=bool)
    for f_index in range(len(faces)):
        round_face_list = [round(pt) for pt in faces[f_index]]
        x_min, y_min, x_max, y_max = np.round(round_face_list)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        if f_index != index:
            mask[y_min:y_max, x_min:x_max] = True
    round_face_list = [round(pt) for pt in faces[index]]
    x_min, y_min, x_max, y_max = np.round(round_face_list)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    mask[y_min:y_max, x_min:x_max] = False
    cloned_grayscale = gray_img.copy()
    cloned_grayscale[mask] = 0
    return cloned_grayscale


if __name__ == '__main__':
    img_path = os.path.join(DATA_ROOT, f'{INPUT_INDEX}.jpg')
    grayscale_path = os.path.join(DATA_ROOT, f'{INPUT_INDEX}.png')
    img = cv2.imread(img_path)
    print(img.shape)
    grayscale_img = cv2.imread(grayscale_path, cv2.IMREAD_GRAYSCALE)
    print(grayscale_img.shape)
    processed_grayscale = remove_redundant(img, grayscale_img)
    cv2.imwrite('./processed.png', processed_grayscale)
