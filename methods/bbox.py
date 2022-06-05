import torch

def convert_bbox(bbox, w, h):
    x1, y1, width, height = bbox[0]
    print(height)

    x1 = x1*w
    y1 = y1*h
    x2 = width*w
    y2 = height*h

    return torch.tensor([x1,y1,x2,y2])
