import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from .align import affineMatrix, landmarks, cropRange, inverseImage

def convert_tensor_shape(input_tensor):
    # Assuming the input tensor shape is (1, 512, 512)
    # and pixel values range from 0 to 18

    # Convert to numpy array
    input_tensor = np.array(input_tensor)

    # Create an empty tensor of the desired shape
    output_tensor = torch.zeros((19, 512, 512))

    # Iterate through each possible pixel value (0-18)
    for i in range(19):
        # Set the corresponding layer in the output tensor
        filtered_tensor = np.where(input_tensor == i, 1, 0)
        output_tensor[i] = torch.from_numpy(filtered_tensor)
    
    return output_tensor

class Dataset(Dataset):
    def __init__(self, root_dir='./data/CelebAMask-HQ', id_file=None, transform=None, unseen=False):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            id_file (string): Path to the file with image IDs for training/testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.unseen = unseen
        with open(id_file, 'r') as file:
            self.image_ids = file.read().splitlines()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.unseen:
            image_path_name = f'{self.image_ids[idx]}.jpg'
            img_name = os.path.join(self.root_dir, '../', 'Unseen', image_path_name)
            cv_image = cv2.imread(img_name, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # cv_image = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # Image alignmnet
            detected_landmarks = landmarks(cv_image, toRGB = True)
            img_mat, img_center = affineMatrix(detected_landmarks)
            r_mat, aligned_size = cropRange(img_mat, cv_image.shape, img_center)
            aligned_img = cv2.warpAffine(cv_image, r_mat, aligned_size)
            aligned_img = cv2.resize(aligned_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(aligned_img)
            
            if self.transform:
                image = self.transform(image)
                
            return image, image, idx, aligned_size, cv_image.shape, r_mat
        else:
            image_path_name = f'{self.image_ids[idx]}.jpg'
            mask_path_name = f'{self.image_ids[idx]}.png'
            splitted_idx = self.image_ids[idx].split('-')
            if len(splitted_idx) > 1:
                if splitted_idx[1] == '1' or splitted_idx[1] == '2' or splitted_idx[1] == '5':
                    mask_path_name = f'{splitted_idx[0]}.png'

            img_name = os.path.join(self.root_dir, 'CelebA-HQ-img', image_path_name)
            mask_name = os.path.join(self.root_dir, 'CelebAMask-HQ-combined_mask', mask_path_name)
            cv_image = cv2.imread(img_name, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image = cv2.resize(cv_image, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # Image alignmnet
            detected_landmarks = landmarks(cv_image, toRGB = True)
            if detected_landmarks is None:
                print(img_name)
                image = Image.open(img_name).convert('RGB').resize((512, 512), Image.BILINEAR)
                mask = Image.open(mask_name).convert('L').resize((512, 512), Image.NEAREST)
                aligned_size = (512, 512)
                r_mat = np.array([[1, 0, 0], [0, 1, 0]])
            else:
                img_mat, img_center = affineMatrix(detected_landmarks)
                r_mat, aligned_size = cropRange(img_mat, cv_image.shape, img_center)
                aligned_img = cv2.warpAffine(cv_image, r_mat, aligned_size)
                aligned_img = cv2.resize(aligned_img, (512, 512), interpolation=cv2.INTER_LINEAR)
                image = Image.fromarray(aligned_img)

                cv_mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
                aligned_mask = cv2.warpAffine(cv_mask, r_mat, aligned_size)
                aligned_mask = cv2.resize(aligned_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                mask = Image.fromarray(aligned_mask)

            # Extend the mask to 19 classes with pixel level
            mask = convert_tensor_shape(mask)

            if self.transform:
                image = self.transform(image)
                # mask = self.transform(mask)

        return image, mask, idx, aligned_size, cv_image.shape, r_mat
