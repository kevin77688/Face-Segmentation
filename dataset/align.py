import cv2
import numpy as np
import os
import math
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch
from math import dist
from scipy.spatial import distance

DATA_ROOT = '/home/kevin/Code/Face-Segmentation/data/CelebAMask-HQ/CelebA-HQ-img/'
OUTPUT_DATA_ROOT = '/home/kevin/Code/Face-Segmentation'
INPUT_IMAGE = '22.jpg'
OUTPUT_IMAGE = 'alignment.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = [0.4, 0.3, 0.3]

def landmarks(img, toRGB=True):
    step = 0.1
    mtcnn = MTCNN(
        thresholds=threshold, 
        device=device
    )
    # if toRGB: 
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while threshold[-1] >= 0:
        try: 
            faces, probs, landmarks = mtcnn.detect(img, landmarks=True)
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
            x_min = max(0, x_min - 50)
            y_min = max(0, y_min - 100)
            x_max = min(w, x_max + 50)
            y_max = min(h, y_max + 100)
            
            mask = np.zeros_like(img[:,:,0], dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            # cloned_image = img.copy()
            img[~mask] = (0, 0, 0)

            landmark = landmarks[index]
            keypoints = {'left_eye': tuple(landmark[0]), 'right_eye': tuple(landmark[1])
                         , 'nose': tuple(landmark[2]), 'mouth_left': tuple(landmark[3]), 'mouth_right': tuple(landmark[4])}
            return keypoints
        except:
            max_index = threshold.index(max(threshold))
            threshold[max_index] -= step
            mtcnn = MTCNN(thresholds=threshold, device=device)

def cropRange(mat, img_shape, center):
    h, w = img_shape[:2]
    corner_point = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    key_point = np.append(corner_point, [center], axis=0)
    homo_key_point = np.column_stack((key_point, np.ones(len(key_point))))
    homo_transformed_point = np.dot(mat, homo_key_point.T).T
    tranformed_center = homo_transformed_point[-1, :2]
    transformed_corner_point = homo_transformed_point[:4, :2]
    w = 0
    for point in transformed_corner_point:
        x, y = point
        w = max(w, abs(tranformed_center[0] - x), abs(tranformed_center[1] - y))
        # w = max(w, abs(tranformed_center[0] - x), abs(tranformed_center[1] - y))
    # w = math.ceil(w)
    translation = np.array([[1, 0, w], 
                            [0, 1, w], 
                            [0, 0, 1]])
    mat = np.dot(translation, mat)
    return mat[:2], (int(2 * w), int(2 * w))

def affineMatrix(lmks, scale=8):
    nose = np.array(lmks['nose'], dtype=np.float32)
    left_eye = np.array(lmks['left_eye'], dtype=np.float32)
    right_eye = np.array(lmks['right_eye'], dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # w = np.sqrt(np.sum(eye_width**2)) * scale
    rotation_matrix = [[alpha, beta, 0],
                       [-beta, alpha, 0],
                       [0, 0, 1]]
    translation_matrix = [[1, 0, -center[0]],
                          [0, 1, -center[1]],
                          [0, 0, 1]]
    m = np.dot(rotation_matrix, translation_matrix)
    return m, center

def inverseImage(aligned_img, aligned_size, original_shape, r_mat):
    resize_img = cv2.resize(aligned_img, aligned_size, interpolation=cv2.INTER_NEAREST)
    r_mat = np.vstack([r_mat, np.array([0, 0, 1])])
    r_mat_inv = np.linalg.inv(r_mat)
    transformed_img = cv2.warpAffine(resize_img, r_mat_inv[:2], (original_shape[1], original_shape[0]))
    return transformed_img

def alignImage(img, toRGB=True):
    mat, center = affineMatrix(landmarks(img, toRGB))
    r_mat, size = cropRange(mat, img.shape, center)
    alignment = cv2.warpAffine(img, r_mat, size)
    aligned_img = cv2.resize(alignment, (512, 512), interpolation=cv2.INTER_LINEAR)
    return aligned_img, size, r_mat

def inverseTensor(aligned_imgs, aligned_predicteds, aligned_masks, aligned_size, original_shape, r_mat, unseen=False):
    batch_size = aligned_imgs.shape[0]
    # for i in aligned_size:
    #     i = i.cpu().numpy()[0]
    # new_aligned_imgs = np.zeros_like([i.cpu().numpy() for i in aligned_size])
    new_aligned_imgs = [_ for _ in range(batch_size)]

    # new_aligned_predicteds = np.empty_like(aligned_size.cpu().numpy())
    new_aligned_predicteds = [_ for _ in range(batch_size)]
    
    new_aligned_masks = np.empty_like(aligned_masks.cpu().numpy())
    
    for b in range(batch_size):
        aligned_img, aligned_predicted, aligned_mask  = aligned_imgs[b], aligned_predicteds[b], aligned_masks[b]
        aligned_img = aligned_img.detach().permute(1, 2, 0).cpu().numpy()
        aligned_predicted = aligned_predicted.detach().cpu().numpy()
        aligned_predicted = aligned_predicted.astype(float)
        aligned_mask = aligned_mask.detach().cpu().numpy()
        
        aligned_size = (aligned_size[1].cpu().numpy()[0], 
                        aligned_size[0].cpu().numpy()[0])
        
        original_shape = (original_shape[1].cpu().numpy()[0],
                            original_shape[0].cpu().numpy()[0],
                            original_shape[2].cpu().numpy()[0])
        r_mat = r_mat.cpu().numpy()[0]
        
        r_mat = np.vstack([r_mat, np.array([0, 0, 1])])
        r_mat_inv = np.linalg.inv(r_mat)
        r_mat_inv = r_mat_inv[:2]
        original_shape = (original_shape[1], original_shape[0])
        
        aligned_img = cv2.resize(aligned_img, aligned_size, interpolation=cv2.INTER_NEAREST)
        aligned_img = cv2.warpAffine(aligned_img, r_mat_inv, original_shape)
        aligned_img = np.transpose(aligned_img, (2, 0, 1))
        
        aligned_predicted = cv2.resize(aligned_predicted, aligned_size, interpolation=cv2.INTER_NEAREST)
        aligned_predicted = cv2.warpAffine(aligned_predicted, r_mat_inv, original_shape)
        
        if not unseen:
            for i in range(19):
                new_aligned_mask = cv2.resize(aligned_mask[i], aligned_size, interpolation=cv2.INTER_NEAREST)
                aligned_mask[i] = cv2.warpAffine(new_aligned_mask, r_mat_inv, original_shape)
            
        new_aligned_imgs[b] = torch.from_numpy(aligned_img)
        new_aligned_predicteds[b] = torch.from_numpy(aligned_predicted)
        
        if not unseen:
            new_aligned_masks[b] = aligned_mask
        
    aligned_imgs = torch.stack(new_aligned_imgs,dim=0).cuda()      #torch.from_numpy(new_aligned_imgs).cuda()
    aligned_predicteds = torch.stack(new_aligned_predicteds,dim=0).cuda()  #torch.from_numpy(new_aligned_predicteds).cuda()
    aligned_masks = torch.from_numpy(new_aligned_masks).cuda()
    return aligned_imgs, aligned_predicteds, aligned_masks

if __name__ == '__main__':
    
    file_path = os.path.join(DATA_ROOT, INPUT_IMAGE)
    img = cv2.imread(file_path)

    # Aligned image
    aligned_img, aligned_size, r_mat = alignImage(img)  # Change to alignImage(img, False) if the input image is RGB
    cv2.imwrite(OUTPUT_IMAGE, aligned_img)

    # Transform back
    transformed_img = inverseImage(aligned_img, aligned_size, (1024, 1024), r_mat)
    cv2.imwrite('transformation.jpg', transformed_img)