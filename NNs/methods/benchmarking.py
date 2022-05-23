import torch 
import torchvision
import torch.nn.functional as F


def IoU_Loss(inputs, targets, smooth=1e-6):

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth)/(union + smooth)

    return 1-IoU
