import os
import numpy as np
from PIL import Image
import cv2

import torch
from torchvision.transforms import transforms
from utils.postprocess import remove_redundant

def combine_masks(input_tensor):
    batchSize, layers, height, width = input_tensor.shape
    output_tensor = torch.full((batchSize, height, width), -1, dtype=torch.long)

    for layer in range(layers):
        mask = (input_tensor[:, layer, :, :] == 1)
        output_tensor[mask] = layer

    return output_tensor.clamp(min=0)

def split_masks(tensor, num_classes=19):
    batch_size = tensor.shape[0]
    masks = torch.zeros((batch_size, num_classes, tensor.shape[-2], tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    for b in range(batch_size):
        for i in range(num_classes):
            masks[b, i] = (tensor[b] == i).type(tensor.dtype)
    return masks
seq = 0
def predict_masks(model, images, csv=True):
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    np_image = images.clone().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
    np_predicted = predicted.detach().cpu().squeeze(0).numpy()
    np_predicted = remove_redundant(np_image, np_predicted)
    predicted = torch.from_numpy(np_predicted).unsqueeze(0).float()
    if csv:
        return torch.nn.functional.interpolate(predicted.unsqueeze(1).float(), size=(256, 256), mode="nearest").cpu()
    else:
        return torch.nn.functional.interpolate(predicted.unsqueeze(1).float(), size=(512, 512), mode="nearest").cpu()

def create_masks_dict(predicted, labels_celeb):
    batch_size = predicted.shape[0]
    dict_data_batch = [{} for _ in range(batch_size)]

    splitted_predict_masks = split_masks(predicted)
    
    for b in range(batch_size):
        for i, label in enumerate(labels_celeb):
            if i == 0:
                dict_data_batch[b][label] = np.zeros((256, 256), dtype=np.uint8)
            else:
                dict_data_batch[b][label] = splitted_predict_masks[b, i].numpy()

    return dict_data_batch

def save_images_if_required(dict_data_batch, idx, labels_celeb, save_images):
    if save_images:
        for b, dict_data in enumerate(dict_data_batch):
            for label in labels_celeb[1:]:  # skip background
                image = transforms.ToPILImage()(torch.tensor(dict_data[label]).squeeze())
                output_dir = f'out/{idx[b].item()}'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                image.save(f'{output_dir}/{label}.png')

def reorder_dict_data(dict_data, labels_celeb_origin):
    return {label: dict_data[label] for label in labels_celeb_origin}