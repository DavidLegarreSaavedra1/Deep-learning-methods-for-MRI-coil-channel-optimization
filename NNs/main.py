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

N_EPOCHS = 1000
BATCH_SIZE = 32
IMG_SIZE = 144
TO_TRAIN = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented' 

    training_ann = root_data_path / 'result.json'

    train_transformers = nn.Sequential(
        T.Resize(size=IMG_SIZE),
        #T.Normalize((0.485),(0.229)),
    )
   
    heart_dataset = ChestHeartDataset(
        root_data_path, training_ann, 
        transforms=train_transformers
    )

    print(len(heart_dataset))

    train_len = int(0.8 * len(heart_dataset))
    test_len = int((len(heart_dataset)-train_len)/2)
    val_len = int(len(heart_dataset)-train_len-test_len)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        heart_dataset, [train_len, val_len, test_len]
    )

    training_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testing_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    net = FastNN(heart_dataset.num_classes())
    net = net.to(device)

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
        epochs, losses, train_losses = train(net, N_EPOCHS, training_data_loader, validate_data_loader, device,root_data_path)

    net.load_state_dict(torch.load(root_data_path / 'net.pth'))
    net.eval()

    test_path = path.cwd() / 'testing'
    testing(test_path, net, device, IMG_SIZE)


    if TO_TRAIN:
        fig, ax = plt.subplots(1,1)
        ax.plot(epochs, losses)
        ax.plot(epochs, train_losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(["Validation loss", "Training loss"])

    show(training_result)

    plt.show()