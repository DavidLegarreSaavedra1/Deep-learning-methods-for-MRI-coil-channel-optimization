from dataset import Dataset
import os
import matplotlib.pyplot as plt

ALL_DATA_DIR =  os.path.join('..', 'input', 'train', 'train')
base_path = os.path.join(ALL_DATA_DIR, '140')
tData = Dataset(base_path,'140')
tData.load()

plt.imshow(tData.images[0,0,:,:], cmap = 'bone')