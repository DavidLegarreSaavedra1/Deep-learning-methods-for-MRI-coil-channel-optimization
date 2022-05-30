from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2 as cv
import json


class ChestHeartDataset(Dataset):
    """Chest MRI image dataset"""

    def __init__(self, root, json_file, transforms=None):
        """ 
        Args:
        root (string): path to the root of the dataset
        jason_file(path): filename of the json containing all the data
        transforms(nn.Sequential): In case some initial transformation of the data is necessary
        """
        self.root = root.as_posix()

        f = open(json_file.as_posix())
        self.json = json.load(f)

        self.images = self.json["images"]
        self.annotations = self.json["annotations"]
        self.transforms = transforms
        self.num_class = len(self.json["categories"])

    def __getitem__(self, index):
        # Get image
        img_path = self.images[index]["file_name"]
        img_path = img_path[:6] + img_path[8:]
        img_path = self.root + '/' + img_path
        img = cv.imread(img_path, 0)
        img = cv.normalize(
            img, None, 0, 255, cv.NORM_MINMAX
        )
        w,h = img.shape
        #img = Image.open(img_path)
        transform = T.Compose([
            T.ToTensor(),
        ])
        img = transform(img)
        if self.transforms:
            img = self.transforms(img)

        # Get annotations
        annotation = self.annotations[index]
        coords = annotation["bbox"]

        x0 = coords[0]
        y0 = coords[1]
        x1 = x0+coords[2]
        y1 = y0+coords[3]
        box = [x0, y0, x1, y1]

        # Define target box
        #bboxes = torch.as_tensor(box, dtype=torch.float32)

        x0 = coords[0]/w
        y0 = coords[1]/h
        x1 = coords[2]/w
        y1 = coords[3]/h
        box = [x0, y0, x1, y1]

        bboxes = torch.as_tensor(box, dtype=torch.float64)
        labels = annotation["category_id"]
        labels = torch.tensor(labels, dtype=torch.int64)

        # return img, target
        return img, bboxes, labels

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return self.num_class
