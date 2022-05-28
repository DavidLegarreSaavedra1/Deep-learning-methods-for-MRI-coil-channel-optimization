from py import process
from methods.neural_network.FastNN import *
from pathlib import Path as path
from methods import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import cv2 as cv

torch.cuda.empty_cache()

N_EPOCHS = 50
BATCH_SIZE = 12
IMG_SIZE = 144
TO_TRAIN = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net = FastNN()
net = net.to(device)

if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented_COCO'

    training_ann = root_data_path / 'train.json'
    testing_ann = root_data_path / 'test.json'
    validate_ann = root_data_path / 'validation.json'

    train_transformers = nn.Sequential(
        T.Resize(size=IMG_SIZE),
        #T.Normalize((0.485),(0.229)),
    )
    validate_transformers = nn.Sequential(
        T.Resize(size=IMG_SIZE),
    )
   
    train_heart_dataset = ChestHeartDataset(
        root_data_path, training_ann, 
        transforms=train_transformers
    )
    test_heart_dataset = ChestHeartDataset(
        root_data_path, testing_ann
    )
    validate_heart_dataset = ChestHeartDataset(
        root_data_path, validate_ann,
        transforms=validate_transformers    
    )
    

    training_data_loader = torch.utils.data.DataLoader(
        train_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testing_data_loader = torch.utils.data.DataLoader(
        test_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        validate_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Draw bounding boxes of training example
    img, bbox, _ = next(iter(testing_data_loader))
    img = cv.normalize(
        img[0].numpy(), None, 
        0, 255, cv.NORM_MINMAX,
        cv.CV_8U
    )
    img = torch.from_numpy(img).to(device)

    training_result = torchvision.utils.draw_bounding_boxes(
        img,
        bbox[0].unsqueeze(0),
        colors='green',
        width=2
    )


    if TO_TRAIN:
        epochs, losses = train(net, N_EPOCHS, training_data_loader, validate_data_loader, device)
        torch.save(net.state_dict(), root_data_path / 'net.pth')

    net.load_state_dict(torch.load(root_data_path / 'net.pth'))
    net.eval()

    test_path = path.cwd() / 'testing'
    testing(test_path, net, device, IMG_SIZE)


    if TO_TRAIN:
        fig, ax = plt.subplots(1,1)
        ax.plot(epochs, losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

    show(training_result)

    plt.show()