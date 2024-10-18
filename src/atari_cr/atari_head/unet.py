""" Code from
https://github.com/milesial/Pytorch-UNet/commit/514f18e1a961fee4bf98fa6f50084a166e3bff4b
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Dropout(dropout),
            torch.compile(nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            torch.compile(nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2,
                                     stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, scale=8, dropout=0.5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, scale, dropout=dropout)
        self.down1 = Down(scale, scale * 2, dropout)
        self.down2 = Down(scale * 2, scale * 4, dropout)
        self.down3 = Down(scale * 4, scale * 8, dropout)
        self.down4 = Down(scale * 8, scale * 16, dropout)
        self.up1 = Up(scale * 16, scale * 8, dropout)
        self.up2 = Up(scale * 8, scale * 4, dropout)
        self.up3 = Up(scale * 4, scale * 2, dropout)
        self.up4 = Up(scale * 2, scale, dropout)
        self.outc = nn.Conv2d(scale, n_classes, kernel_size=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.log_softmax(x)
        x = x.view(x.size(0), 84, 84)

        return x

if __name__ == "__main__":
    net = UNet(4, 1)
    print(sum([p.numel() for p in net.parameters()]))
