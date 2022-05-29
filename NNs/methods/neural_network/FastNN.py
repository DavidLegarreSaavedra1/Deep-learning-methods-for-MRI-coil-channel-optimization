import enum
from turtle import forward
from matplotlib.pyplot import box
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=0.25)

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

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class FastNN(nn.Module):
    # Input images of size 512 512
    def __init__(self, num_classes):
        super(FastNN, self).__init__()
        # CNN phase
        self.conv1 = BasicConv2d(1, 32, kernel_size=3)
        self.conv2 = BasicConv2d(32, 64, kernel_size=3)
        self.conv3 = BasicConv2d(64, 128, kernel_size=3)
        self.conv4 = BasicConv2d(128, 256, kernel_size=3)
        self.conv5 = BasicConv2d(256, 512, kernel_size=3)

        # To distinguish the slice of the heart
        self.class1 = BasicFc(512 * 2 * 2, 256)
        self.class2 = BasicFc(256, 128)
        self.class3 = BasicFc(128, 64)
        self.class4 = BasicFc(64, 32)
        self.class_ = nn.Linear(32, num_classes)

        # To predict bounding boxes
        self.boxc1 = BasicFc(512 * 2 * 2, 256)
        self.boxc2 = BasicFc(256, 128)
        self.boxc3 = BasicFc(128, 64)
        self.boxc4 = BasicFc(64, 32)
        self.box = nn.Linear(32, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)

        class_pred = self.class1(x)
        class_pred = self.class2(class_pred)
        class_pred = self.class3(class_pred)
        class_pred = self.class4(class_pred)
        class_pred = self.class_(class_pred)
        class_pred = torch.sigmoid(class_pred)

        box_pred = self.boxc1(x)
        box_pred = self.boxc2(box_pred)
        box_pred = self.boxc3(box_pred)
        box_pred = self.boxc4(box_pred)
        box_pred = self.box(box_pred)
        box_pred = nn.Sigmoid()(box_pred)

        return class_pred, box_pred
