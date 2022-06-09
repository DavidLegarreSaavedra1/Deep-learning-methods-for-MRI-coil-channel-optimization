import enum
from turtle import forward
from matplotlib.pyplot import box
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import time


class TestNN(nn.Module):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.75)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.75)
        )
        self.box_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(10*21*21, 4),
        )
        self.resize = T.Resize(image_size)

    def forward(self, x: torch.Tensor):
        x = self.resize(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.box_regressor(x)

        return x

class FastNN(nn.Module):
    def __init__(self, image_size) -> None:
        super().__init__()
        self.resize = T.Resize(image_size)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.box_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10*46*46, 4)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.resize(x)

        x = self.block1(x)
        x = self.box_regressor(x)

        return x