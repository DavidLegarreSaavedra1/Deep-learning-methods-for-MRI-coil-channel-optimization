import enum
from turtle import forward
from matplotlib.pyplot import box
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import time


class FastNN(nn.Module):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.75)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(10, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.box_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(24*12*12, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 4),
                nn.Dropout(0.6),
                nn.LeakyReLU(),
        )
        self.resize = T.Resize(image_size)

    def forward(self, x: torch.Tensor):
        x = self.resize(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.box_regressor(x)

        return x

class TestNN(nn.Module):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.resize = T.Resize(image_size)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(10, 24, 3, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.box_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*5*5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
            #nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.resize(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.box_regressor(x)

        return x