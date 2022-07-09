from ROVir import *
import nibabel as nib
import numpy as np

def extract_coils(coils_path):
    nib_obj = nib.load(coils_path)
    img_np = np.array(nib_obj.dataobj) 
    img_np = np.flip(img_np, [0,1])
    return img_np

def get_avg_intensity(region, image):
    return np.mean(image[region[0], region[1]])

def auto_drop_coils(img, coils, ROI):
    """Automatic last coil to drop generation
    """

    print(ROI[0])

    limit_intensity = 0.7*get_avg_intensity(ROI, img)

    ncoils = coils.shape[-1]

    print(limit_intensity)
    for i in range(1, ncoils):
        print(i)
        test = coils[...,:i]
        print(test.shape[2])
        coils_i = combine_images(coils[...,:-i])
        intensity_i = get_avg_intensity(ROI, coils_i)
        print(intensity_i)

        if intensity_i <= limit_intensity:
            return coils_i
    
    return coils_i


    


