from struct import unpack
from telnetlib import XASCII
import numpy as np
import torch
import torchvision
import torch.nn as nn

def unpack_coords(box):
    coords = []
    for i in range(4):
        coords.append(
            box[:, i]
        )
    x1,y1,x2,y2 = coords
    x1 = x1.unsqueeze(1)
    y1 = y1.unsqueeze(1)
    x2 = x2.unsqueeze(1)
    y2 = y2.unsqueeze(1)

    return x1,y1,x2,y2

def IoU_index(pred, target, smooth=1e-6):
    pred = pred*512
    target = target*512
    px1, py1, px2, py2 = unpack_coords(pred)
    tx1, ty1, tx2, ty2 = unpack_coords(target)

    x1 = torch.max(px1, tx1)
    y1 = torch.max(py1, ty1)
    x2 = torch.min(px2, tx2)
    y2 = torch.min(py2, ty2)

    intersection = (x2 - x1).clamp(0) * (y2-y1).clamp(0)

    pred_area = (px2 - px1) * (py2 - py1)
    targ_area = (tx2 - tx1) * (ty2 - ty1)


    return (intersection + smooth) / (pred_area+targ_area - intersection + smooth)


def IoU_loss(pred, target):
    #target /= 512
    pred = pred * 512
    #index = 1-IoU_index(pred, target).mean()
    #print(f"{pred[0]=}")
    #print(f"{target[0]=}")
    index = 1 - IoU_index(pred, target).mean()
    #print(f"{index=}")
    loss = nn.L1Loss()(pred, target)
    loss -= index

    return loss