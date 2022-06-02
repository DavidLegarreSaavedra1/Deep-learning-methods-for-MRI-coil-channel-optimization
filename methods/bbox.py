import numpy as np

def convert_bbox(bbox, w, h):
    x1, y1, width, height = bbox

    x1 = x1*w
    y1 = y1*h
    x2 = x1+width*w
    y2 = y1+height*h

    return x1,y1,x2,y2
