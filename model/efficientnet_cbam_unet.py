import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models
from .cbam import CBAM

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_rate=0.2):
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
    def __init__(self, n_channels=3, n_classes=19, encoder_name='efficientnet-b4'):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained(encoder_name)

        self.decoder1 = DecoderBlock(1792, 1792, 448)
        self.decoder2 = DecoderBlock(448, 448, 160)
        self.decoder3 = DecoderBlock(160, 160, 56)
        self.decoder4 = DecoderBlock(56, 56, 32)
        self.decoder5 = DecoderBlock(32, 32, 24)

        self.cbam_final = CBAM(24)

        self.out_conv = nn.Conv2d(24, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder.extract_features(x)

        # Decoder
        x = self.decoder1(features)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder5(x)

        # CBAM
        x = self.cbam_final(x)

        # Output
        x = self.out_conv(x)
        return x
