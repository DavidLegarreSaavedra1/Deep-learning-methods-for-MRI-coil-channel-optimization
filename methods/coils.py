import nibabel as nib
import numpy as np

def extract_coils(coils_path):
    nib_obj = nib.load(coils_path)
    img_np = np.array(nib_obj.dataobj) 
    img_np = np.flip(img_np, [0,1])
    return np