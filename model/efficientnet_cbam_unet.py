import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models
from .cbam import CBAM

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

class Model(nn.Module):
    def __init__(self, n_channels=3, n_classes=19, encoder_name='efficientnet-b7'):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained(encoder_name)

        self.decoder1 = DecoderBlock(2560 + 640, 2560, 640)
        self.decoder2 = DecoderBlock(640 + 224, 640, 224)
        self.decoder3 = DecoderBlock(224 + 80, 224, 80)
        self.decoder4 = DecoderBlock(80 + 48, 80, 48)
        self.decoder5 = DecoderBlock(48 + 32, 48, 32)
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # self.decoder1 = DecoderBlock(1792 + 448, 1792, 448)
        # self.decoder2 = DecoderBlock(448 + 160, 448, 160)
        # self.decoder3 = DecoderBlock(160 + 56, 160, 56)
        # self.decoder4 = DecoderBlock(56 + 32, 56, 32)
        # self.decoder5 = DecoderBlock(32 + 24, 32, 24)
        # self.out_conv = nn.Conv2d(24, n_classes, kernel_size=1)
        

    def forward(self, x):
        # Encoder
        features = self.encoder.extract_features(x)
        endpoints = self.encoder.extract_endpoints(x)
        reduction_1 = endpoints['reduction_1']  # torch.Size([1, 32, 112, 112])
        reduction_2 = endpoints['reduction_2']  # torch.Size([1, 48, 56, 56])
        reduction_3 = endpoints['reduction_3']  # torch.Size([1, 80, 28, 28])
        reduction_4 = endpoints['reduction_4']  # torch.Size([1, 224, 14, 14])
        reduction_5 = endpoints['reduction_5']  # torch.Size([1, 640, 7, 7])
        # reduction_6 = endpoints['reduction_6']  # torch.Size([1, 2560, 7, 7])

        # Decoder
        x = self.decoder1(torch.cat([features, reduction_5], dim=1))
        x = self.decoder2(torch.cat([x, reduction_4], dim=1))
        x = self.decoder3(torch.cat([x, reduction_3], dim=1))
        x = self.decoder4(torch.cat([x, reduction_2], dim=1))
        x = self.decoder5(torch.cat([x, reduction_1], dim=1))
        
        # Output
        x = self.out_conv(x)
        return x
