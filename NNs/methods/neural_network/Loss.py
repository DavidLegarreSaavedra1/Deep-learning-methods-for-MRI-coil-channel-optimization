from telnetlib import XASCII
import numpy as np
import torch
import torchvision


def intersection(a, b):
    dx = np.min(a[0], b[0]) - np.max(a[2], b[2])
    dy = np.min(a[1], b[1]) - np.max(a[3], b[3])
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return 0


def area_rect(a):
    xside = a[0] - a[2]
    yside = a[1] - b[3]
    return xside*yside


def my_IoU(pred, target, smooth=1e-7):
    intersec = intersection(pred, target)

    area_pred = area_rect(pred)
    are_target = area_rect(target)

def bb_intersection_over_union(boxA, boxB, smooth=1e-6):
    range_batches = torch.arange(boxA.shape[0])
    xA = torch.maximum(boxA[range_batches,0], boxB[range_batches,0])
    yA = torch.maximum(boxA[range_batches,1], boxB[range_batches,1])
    xB = torch.minimum(boxA[range_batches,2], boxB[range_batches,2])
    yB = torch.minimum(boxA[range_batches,3], boxB[range_batches,3])


    xdif = xB-xA
    ydif = yB-yA


    if torch.all(xdif >= 0) and torch.all(ydif >= 0):
        interArea = xdif*ydif
    else:
        interArea = 0
    
    boxA_area = (boxA[2] - boxA[0]+smooth) * (boxA[3]-boxA[1]+smooth)
    boxB_area = (boxB[2] - boxB[0]+smooth) * (boxB[3]-boxB[1]+smooth)


    iou = (interArea+smooth) / (boxA_area+boxB_area - interArea).float()

    return iou


