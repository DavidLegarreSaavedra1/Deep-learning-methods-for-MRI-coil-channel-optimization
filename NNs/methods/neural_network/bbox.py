
def convert_bbox(bbox, w, h):
    x1,y1,x2,y2 = bbox

    x1 = x1*w
    y1 = y1*h
    x2 = x2*w
    y2 = y2*h

    return x1,y1,x2,y2