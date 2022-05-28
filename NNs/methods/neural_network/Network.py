from pickletools import optimize
from matplotlib.pyplot import box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # CNNs for rgb images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)


        # Connecting CNN outputs with Fully Connected layers for bounding box
        #self.box_fc1 = nn.Linear(in_features=192 * 11 * 11, out_features=240)
        self.box_fc1 = nn.Linear(in_features=192, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_fc3 = nn.Linear(in_features=120, out_features=60)
        self.box_out = nn.Linear(in_features=60, out_features=4)

        self.dropout = nn.Dropout(.25)
        self.lrn  = nn.LocalResponseNorm(2)
        self.ln = nn.LayerNorm(121)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)




    def forward(self, t):
        t = self.conv1(t)
        t = self.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = self.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = self.relu(t)
        t = self.ln(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = self.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout(t)

        t = self.conv5(t)
        t = self.relu(t)
        t = self.lrn(t)
        t = self.avgpool(t)

        t = torch.flatten(t,start_dim=1)
        
        box_t = self.box_fc1(t)
        box_t = self.relu(box_t)

        box_t = self.dropout(box_t)
        box_t = self.box_fc2(box_t)
        box_t = self.relu(box_t)

        box_t = self.box_fc3(box_t)
        box_t = self.relu(box_t)

        box_t = self.box_out(box_t)
        box_t = torch.sigmoid(box_t)

        return box_t

