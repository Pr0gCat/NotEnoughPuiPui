import torch
import torch.nn as nn

class NEPPDiscriminator(nn.Module):
    """
        輸入1張3x64x64的圖
        輸出圖片有多真
    """
    def __init__(self):
        super(NEPPDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



class NEPPGenerator(nn.Module):
    """
        輸入100個隨機數
        輸出一張3通道(RGB)的圖
    """

    def __init__(self):
        super(NEPPGenerator, self).__init__()
        self.net_stack = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #in: 512x4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #in: 256x8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            #in: 128x16x16
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            #in: 64x32x32
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net_stack(x)