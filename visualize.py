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

def visualize_predictions(image, prediction):
    color_map = draw_segmentation_map(prediction).cpu().numpy()
    image = image.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.transpose(image[0], (1, 2, 0)))  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(np.transpose(color_map[0], (1, 2, 0)))  # Same channel rearrangement
    axes[1].set_title('Segmentation Map')
    axes[1].axis('off')
    
    plt.savefig('/home/kevin/Code/CV_Workshop/Face_Competition/out/test.png')