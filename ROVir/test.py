from methods import *
from pathlib import Path as path
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom


DATA_PATH = path.cwd() / 'data' / 'dicom_images'
IMG_RANGE = range(47,85)

def get_number(string):
    num = int(string[-7:-4])
    return num

if __name__ == '__main__':
    
    for img in DATA_PATH.iterdir():
        if img.is_file():
            num = get_number(img.as_posix())
            if num in IMG_RANGE:
                print("img found at", num)
                data = dicom.dcmread(img)
                img_dcm = Image.fromarray(data.pixel_array)
                img_dcm.save(img.as_posix()+".png")