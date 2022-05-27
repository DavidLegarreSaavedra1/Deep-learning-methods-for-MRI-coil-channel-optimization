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


class ChestHeartDataset(Dataset):
    """Chest MRI image dataset"""

    def __init__(self, root, annotation_file):
        """ Args:

        root (string): path to the root of the dataset
        filenames (string): filenames of the images
        annotation_file (string): path to the JSON annotation file
        """
        self.root = root
        #self.annotation_file = annotation_file
        self.coco_annotation = COCO(annotation_file=annotation_file)

        # Only 1 category, heart, 0 would be background
        self.img_ids = self.coco_annotation.getImgIds(catIds=1)


    def __getitem__(self, index):
        # Get image
        img_id = self.img_ids[index]
        img_info = self.coco_annotation.loadImgs([img_id])[0]['file_name']
        #img = Image.open(os.path.join(self.root, img_info))
        img = cv.imread(os.path.join(self.root, img_info), 0)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        m,s = np.mean(img), np.std(img)
        preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=m, std=s),
        ])
        img = preprocess(img)
        #img = torch.permute(img, (0,3,1,2)).float()

        # Get annotations
        ann_ids = self.coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco_annotation.loadAnns(ann_ids)[0]
        coords = anns["bbox"]
        x0 = coords[0]
        y0 = coords[1]
        x1 = x0+coords[2]
        y1 = y0+coords[3]
        box = [x0, y0, x1, y1]

        # Define target box
        bboxes = torch.as_tensor(box, dtype=torch.float)
        #bboxes = torch.as_tensor(coords, dtype=torch.float32)
        labels = len(self.coco_annotation.loadAnns(ann_ids))
        labels = torch.ones((labels), dtype=torch.int64)

        #return img, target
        return img, bboxes, labels

    def __len__(self):
        return len(self.img_ids)
