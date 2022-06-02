import enum
from turtle import forward
from matplotlib.pyplot import box
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import time


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class BasicFc(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class FastNN(nn.Module):
    # Input images of size
    def __init__(self, image_size):
        super(FastNN, self).__init__()
        # CNN phase
        self.conv1 = BasicConv2d(1, 32, kernel_size=3)
        self.conv2 = BasicConv2d(32, 64, kernel_size=3)
        self.conv3 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = BasicConv2d(256, 512, kernel_size=3, padding=1)

        # To predict bounding boxes
        #self.boxc1 = BasicFc(512 * 6 * 6, 256)
        self.boxc1 = BasicFc(2048, 256)
        self.boxc2 = BasicFc(256, 128)
        self.boxc3 = BasicFc(128, 64)
        self.boxc4 = BasicFc(64, 32)
        self.box = nn.Linear(32, 4)
        self.resize = T.Resize(image_size)

    def forward(self, x):
        x = self.resize(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)

        box_pred = self.boxc1(x)
        box_pred = self.boxc2(box_pred)
        box_pred = self.boxc3(box_pred)
        box_pred = self.boxc4(box_pred)
        box_pred = self.box(box_pred)
        #box_pred = nn.Sigmoid()(box_pred)

        return box_pred
