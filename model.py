import os
import argparse
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ------------------ UNet ------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_ch = in_ch
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        decoder_in_ch = features[-1]*2

        # Decoder
        self.ups = nn.ModuleList()
        rev_feats = list(reversed(features))
        for feat in rev_feats:
            self.ups.append(nn.ConvTranspose2d(decoder_in_ch, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(decoder_in_ch, feat))  # input to DoubleConv is skip + upsampled = feat*2
            decoder_in_ch = feat

        # Final
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # upsample
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = T.functional.resize(x, skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)  # DoubleConv

        return self.final_conv(x)
    
    def get_features(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = T.functional.resize(x, skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)
        # Before final_conv
        return x  # (B, features, H, W)


