import torch
import matplotlib.pyplot as plt
import numpy as np

colors = torch.tensor([
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230],
    [119, 11, 32]
], dtype=torch.float32) / 255.0 
    

def draw_segmentation_map(predictions):
    device = predictions.device
    colors_on_device = colors.to(device)

    # Convert the prediction to color image
    color_predictions = colors_on_device[predictions]
    return color_predictions.permute(0, 3, 1, 2)  # Rearrange to [N, C, H, W]

def compress_masks(masks):
    # Create an empty tensor for the compressed masks
    # Assuming masks is of shape [N, C, H, W], the new shape will be [N, H, W]
    compressed = torch.full(masks.shape[1:], -1, dtype=torch.long, device=masks.device)

    # Iterate over each channel
    for channel in range(masks.shape[1]):
        # Wherever the mask is 1 in the current channel, set that value in the compressed mask
        compressed[(masks[:, channel] == 1) & (compressed == -1)] = channel

    return compressed

def visualize_predictions(image, prediction, gt, path='out/test.png'):
    color_map = draw_segmentation_map(prediction).cpu().numpy()
    image = image.cpu().numpy()
    gt = compress_masks(gt.cpu())
    color_map_gt = draw_segmentation_map(gt).cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(np.transpose(image[0], (1, 2, 0)))  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(np.transpose(color_map[0], (1, 2, 0)))  # Same channel rearrangement
    axes[1].set_title('Segmentation Map')
    axes[1].axis('off')

    axes[2].imshow(np.transpose(color_map_gt[0], (1, 2, 0)))  # Same channel rearrangement
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.savefig(path)
    return fig