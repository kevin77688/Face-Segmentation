from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model.unet import Model
from dataset.dataset import Dataset
from visualize import visualize_predictions

# Freeze seed
torch.manual_seed(0)

# Change current working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
MODE = 'train'

# Create U-Net model
model = Model(3, 19)
model = model.to(memory_format=torch.channels_last)
model = model.to(device)

# Print total number of parameters
print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load dataset
train_dataset = Dataset(id_file='/home/kevin/Code/CV_Workshop/Face_Competition/data/train_toy_idx.txt', transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = Dataset(id_file='/home/kevin/Code/CV_Workshop/Face_Competition/data/test_toy_idx.txt', transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
if MODE == 'train':
    lowest_test_loss = 1e10
    model.train()
    for epoch in tqdm(range(EPOCHS), leave=True):
        for images, masks in tqdm(train_loader, leave=False):
            images = images.to(device)
            masks = masks.to(device).squeeze(1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            sys.exit()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for images, masks in tqdm(test_loader, leave=False):
                images = images.to(device)
                masks = masks.to(device).squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, masks)
                if loss < lowest_test_loss:
                    lowest_test_loss = loss
                    torch.save(model.state_dict(), 'checkpoint/unet_model.pth')
                _, predicted = torch.max(outputs.data, 1)

                total += masks.size(0) * masks.size(1) * masks.size(2) * masks.size(3)
                correct += (predicted == masks).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Test Accuracy: {(100 * correct / total):.4f}%, Test Loss: {loss.item():.4f}')

elif MODE == 'test':
    model.load_state_dict(torch.load('checkpoint/unet_model.pth'))
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(test_loader, leave=False):
            images = images.to(device)
            masks = masks.to(device).squeeze(1)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(set(predicted.flatten().cpu().numpy()))

            visualize_predictions(images, predicted)
            break