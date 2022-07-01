from NNs import *
from ROVir import *
from methods import *
from pathlib import Path as path
import cv2 as cv
import torchvision.transforms as T

WEIGHTS_PATH = (path.cwd() / "model" / "net.pth").as_posix()
IMG_PATH = (path.cwd() / "data" / "input.png").as_posix()
COILS_PATH = (path.cwd() / "data")
try:
    COILS_PATH = list(COILS_PATH.glob('*.nii'))[0].as_posix()
except:
    print("\nError: No NIFTI  file found\n")
    exit()

IMG_SIZE = 128

coils = extract_coils(COILS_PATH)
model = load_model(WEIGHTS_PATH, IMG_SIZE)

w, h = coils.shape[:2]

prev_img = combine_images(coils)


# Preprocess image to pass through the network
img_ = preprocess(prev_img)
# Use our model to obtian the ROI
ROI = model(img_)
# postprocess the results
img, ROI = postprocess(prev_img, ROI)


# Get coordinates of the ROI
x1, y1, x2, y2 = np.rint(ROI.detach().numpy()[0]).astype(int)


A_W = slice(x1, x2)
A_H = slice(y1, y2)
B_H = slice(80, 450)
B1_W = slice(0, int(.2*w))
B2_W = slice(int(0.8*w), -1)
regions = [A_W, A_H, B1_W, B_H, B2_W]

rovir_coils, _ = ROVir(coils, regions)

new_img = combine_images(rovir_coils)
new_img = cv.normalize(
    new_img, None, 0, 255,
    cv.NORM_MINMAX
)

plot_images(
    prev_img,
    "Before ROVir",
    255,
    new_img,
    "After ROVir",
    255
)

# closing all open windows
plt.show()
