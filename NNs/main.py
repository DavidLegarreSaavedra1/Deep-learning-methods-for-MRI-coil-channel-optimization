from py import process
from methods.neural_network.FastNN import *
from pathlib import Path as path
from methods import *
from matplotlib import style
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import cv2 as cv

style.use('ggplot')
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams["savefig.bbox"] = "tight"
torch.cuda.empty_cache()


N_EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 96
TO_TRAIN = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' 

    training_ann = root_data_path / 'result.json'

    train_transformers = nn.Sequential(
        T.Resize(size=IMG_SIZE),
        #T.Normalize((0.485),(0.229)),
    )
   
    heart_dataset = ChestHeartDataset(
        root_data_path, training_ann, 
        transforms=train_transformers
    )


    train_len = int(0.7 * len(heart_dataset))
    test_len = int((len(heart_dataset)-train_len)/2)
    val_len = int(len(heart_dataset)-train_len-test_len)

    train_dataset, val_dataset= torch.utils.data.random_split(
        heart_dataset, [train_len, val_len*2]
    )

    training_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    #testing_data_loader = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=BATCH_SIZE,
    #    shuffle=True,
    #    pin_memory=True,
    #)
    validate_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    
    net = FastNN(IMG_SIZE)
    #net = TestNN(IMG_SIZE)
    net = net.to(device)

    if TO_TRAIN:
        epochs, losses, train_losses, saved_loss, best_epoch = train(net, N_EPOCHS,
                training_data_loader, validate_data_loader, 
                device,root_data_path, IMG_SIZE
        )

    net.load_state_dict(torch.load(root_data_path / 'net.pth'))
    net.eval()

    test_path = path.cwd() / 'testing'
    testing(test_path, net, device, IMG_SIZE)


    if TO_TRAIN:
        fig, ax = plt.subplots(1,1)
        ax.plot(epochs, losses)
        ax.plot(epochs, train_losses)
        ax.scatter(best_epoch, saved_loss, c='g')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(["Validation loss", "Training loss", "Model saved epochs"])


    plt.show()
