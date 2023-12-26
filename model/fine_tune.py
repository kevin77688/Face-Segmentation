import torch
import torch.nn as nn
import copy
from efficientnet_pytorch import EfficientNet
from torchvision import models
from .cbam import CBAM
from .efficientnet_cbam_unet import Model as m

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_rate=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(middle_channels),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels),
            nn.Dropout2d(dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.block(x)


class Model(m):
    def __init__(self, n_channels=3, n_classes=19, encoder_name='efficientnet-b7'):
        super().__init__(n_channels, n_classes, encoder_name)
        super().load_state_dict(torch.load('/home/kevin/Code/Face-Segmentation/checkpoint/2023_12_14_135413_EfficientNet_CBAM_UNet/EfficientNet_CBAM_UNet.pth'))
        for param in super().parameters():
            param.requires_grad = False
            
        self.finetune_decoder5 = DecoderBlock(48 + 32, 48, 32)
        for param in self.finetune_decoder5.parameters():
            param.requires_grad = True
            param.data.zero_()
        

    def forward(self, x):
        # Encoder
        features = self.encoder.extract_features(x)
        endpoints = self.encoder.extract_endpoints(x)
        reduction_1 = endpoints['reduction_1']                              # torch.Size([1, 32, 112, 112])
        reduction_2 = endpoints['reduction_2']                              # torch.Size([1, 48, 56, 56])
        reduction_3 = endpoints['reduction_3']                              # torch.Size([1, 80, 28, 28])
        reduction_4 = endpoints['reduction_4']                              # torch.Size([1, 224, 14, 14])
        reduction_5 = endpoints['reduction_5']                              # torch.Size([1, 640, 7, 7])
        # reduction_6 = endpoints['reduction_6']  # torch.Size([1, 2560, 7, 7])

        # Decoder
        x = self.decoder1(torch.cat([features, reduction_5], dim=1))
        x = self.decoder2(torch.cat([x, reduction_4], dim=1))
        x = self.decoder3(torch.cat([x, reduction_3], dim=1))
        x = self.decoder4(torch.cat([x, reduction_2], dim=1))
        x = self.decoder5(torch.cat([x, reduction_1], dim=1)) + self.finetune_decoder5(torch.cat([x, reduction_1], dim=1))
        
        # Output
        x = self.out_conv(x)
        return x
        