import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from typing import List


class ShortcutBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = ShortcutBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))
        return self.act2(x + shortcut)


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, bottle_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottle_channels, kernel_size=1, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(bottle_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(bottle_channels, bottle_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(bottle_channels)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(bottle_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Identity()
        if stride != 1 or (in_channels != out_channels):
            self.shortcut = ShortcutBlock(in_channels, out_channels, stride=stride)

    def forward(self, x:torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + shortcut)
    
    
class ResNet(nn.Module):
    def __init__(self, num_blocks: List[int], strides : List[int], channels: List[int], first_kernel_size: int = 7, image_channels: int = 3, n_classes=10, bottleneck: List[int] = None):
        """
        MNIST 크기 (28 * 28)에 맞춰주기 위해 stride 크기 조정
        """
        super().__init__()
        self.firstconv = nn.Conv2d(image_channels, channels[0], kernel_size=first_kernel_size, stride=strides[0], padding=first_kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bottleneck = bottleneck
        
        self.layers = []
        for i in range(1, len(channels)):
            if self.bottleneck:
                self.layers += self._repeat_bottleneck_layers(channels[i - 1], bottleneck[i - 1], channels[i], stride=strides[i], n_count=num_blocks[i - 1])
            else:
                self.layers += self._repeat_layers(channels[i - 1], channels[i], stride=strides[i], n_count=num_blocks[i - 1])

        self.layers = nn.Sequential(*self.layers)
        self.fc = nn.Linear(channels[i], n_classes)
    

    def _repeat_bottleneck_layers(self, in_channels, bottle_channels, out_channels, stride, n_count):
        layers = []
        strides = [stride] + [1] * (n_count - 1)

        for i, s in enumerate(strides):
            if i == 0:
                layers.append(BottleNeckBlock(in_channels, bottle_channels, out_channels, s))
            else:
                layers.append(BottleNeckBlock(out_channels, bottle_channels, out_channels, s))
                
        return layers
    

    def _repeat_layers(self, in_channels, out_channels, stride, n_count):
        layers = []
        strides = [stride] + [1] * (n_count - 1)
        for i, s in enumerate(strides):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, s))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, s))

        return layers


    def forward(self, x):
        x = self.act1(self.bn1(self.firstconv(x)))
        x = self.maxpool1(x)
        x = self.layers(x)
        x = x.view(x.size(0), x.size(1), -1) # 2차원으로 만들어주기
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x

def resnet18_for_mnist():
    return ResNet(num_blocks=[2, 2, 2, 2], strides=[1, 1, 2, 2, 2], channels=[64, 64, 128, 256, 512], image_channels=1)