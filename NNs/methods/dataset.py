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


def auto_contrast(image, q=.99, dim=None):
    flat = image.flatten()
    sort_flat = np.sort(flat)
    nval = len(image[image < 50])
    limit = np.quantile(sort_flat[nval:], q)
    flat[flat > limit] = limit
    flat[flat < 100] = 100
    if dim:
        return flat.reshape(*image.shape[:2], dim)
    return flat.reshape(*image.shape[:2])

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
        #img = auto_contrast(img)
        img = img.astype(np.float)
        w,h = img.shape[:2]
        #_, bins = np.histogram(img.flatten(), bins=64)
        #limit = np.quantile(bins, 0.8)
        #img /= limit
        #img = Image.open(img_path)
        tensorizer = T.Compose([
            T.ToTensor(),
        ])
        img = tensorizer(img)
        img = img.float()
        #img = (img-img.min())/img.max()
        img = T.Normalize(
            mean=img.mean(),
            std=img.std()
        )(img)
        #img = img.type(torch.float)

        # Get annotations
        annotation = self.annotations[index]
        coords = annotation["bbox"]

        x0 = coords[0]/w
        y0 = coords[1]/h
        x1 = x0+coords[2]/w
        y1 = y0+coords[3]/h
        box = [x0, y0, x1, y1]

        # Define target box
        bboxes = torch.as_tensor(box, dtype=torch.float)

        # return img, target
        return img, bboxes

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return self.num_class
