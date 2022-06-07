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
            nn.Conv2d(1, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.box_regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(10*24*24, 1024),
                nn.ReLU(),
                nn.Linear(1024, 4),
        )
        self.resize = T.Resize(image_size)

    def forward(self, x: torch.Tensor):
        x = self.resize(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.box_regressor(x)

        return x
