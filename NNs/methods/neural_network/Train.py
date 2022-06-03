from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from logging import root
from datetime import datetime
from timeit import default_timer as timer 
from .Loss import *
from .bbox import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import torchvision
import time


def loss_fn(pred, target):
    #target /= 144
    # print(f"{pred=}")
    # print(f"{target=}")
    loss = nn.SmoothL1Loss()(pred, target)
    #loss = nn.L1Loss()(pred, target)
    return loss

def train_step(
    model, loss_fn, optimizer, 
    image, bbox
):
    model.train()
    bbox_pred = model(image)

    loss = loss_fn(bbox_pred, bbox)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def val_step(
    model, loss_fn, 
    image, bbox
):
    model.eval()
    with torch.inference_mode():
        val_pred = model(image)
        val_loss = loss_fn(val_pred, bbox)

    return val_loss

def train(
    model, n_epochs, train_dataloader,
    valdataloader, device, root_data_path,
    img_size
):
    train_start = timer()
    epochs = []
    val_losses = []
    train_losses = []

    optimizer = torch.optim.SGD(
            model.parameters(), lr=1e-2
    )

    for epoch in tqdm(range(n_epochs)):
        print(f"\nEpoch: {epoch}\n----",end="\r")
        train_loss = 0
        val_loss = 0

        for batch, (image, bbox) in enumerate(train_dataloader):
            image,bbox = image.to(device), bbox.to(device)
            train_loss += train_step(
                    model, loss_fn, optimizer,
                    image, bbox
            ).item()          

            val_loss += val_step(
                    model, loss_fn,
                    image, bbox
            ).item()

        train_loss /= len(train_dataloader)
        val_loss /= len(valdataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}")

        epochs.append(epoch)
        #val_losses.append(val_loss.cpu().detach().numpy())
        #train_losses.append(train_loss.cpu().detach().numpy())
        val_losses.append(val_loss)
        train_losses.append(train_loss)

    train_finish = timer()

    print(f"Train time on {device}: {train_finish-train_start:.3f} seconds")

    return epochs[1:], val_losses[1:], train_losses[1:]

