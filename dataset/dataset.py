import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
    def __init__(self, root_dir='./data/CelebAMask-HQ', id_file=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            id_file (string): Path to the file with image IDs for training/testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        with open(id_file, 'r') as file:
            self.image_ids = file.read().splitlines()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'CelebA-HQ-img', f'{self.image_ids[idx]}.jpg')
        image = Image.open(img_name).convert('RGB').resize((512, 512))

        mask_name = os.path.join(self.root_dir, 'CelebAMask-HQ-combined_mask', f'{self.image_ids[idx]}.png')
        mask = Image.open(mask_name).resize((512, 512))

        # Extend the mask to 19 classes with pixel level
        mask = convert_tensor_shape(mask)

        if self.transform:
            image = self.transform(image)
            # mask = self.transform(mask)

        return image, mask
