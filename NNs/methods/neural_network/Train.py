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
    #loss = nn.SmoothL1Loss()(pred, target)
    loss = nn.L1Loss()(pred, target)
    #loss = nn.MSELoss()(pred, target)
    return loss


def train_step(
    model, loss_fn, optimizer,
    data_loader, device
):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    return train_loss


def val_step(
    model, loss_fn,
    data_loader, device
):
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)

        test_loss /= len(data_loader)

    return test_loss


def train(
    model, n_epochs, train_dataloader,
    valdataloader, device, root_data_path,
    img_size
):
    train_start = timer()
    epochs = []
    val_losses = []
    train_losses = []
    loss_saved = []
    best_epoch = []
    best_vloss = 1_000_000
    patience = 7
    trigger = 0

    adam = False

    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-3,
        momentum=0.9, weight_decay=1e-1
    )

    if adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-2
        )

    for epoch in tqdm(range(n_epochs)):
        print(f"\nEpoch: {epoch}\n----")
        train_loss = train_step(
            model, loss_fn, optimizer,
            train_dataloader, device 
        ).item()

        val_loss = val_step(
            model, loss_fn,
            valdataloader, device
        ).item()

        print(f"\nTrain loss: {train_loss:.5f} | Validation loss: {val_loss:.5f}")

        if val_loss < best_vloss:
            best_vloss = val_loss
            trigger = 0
            loss_saved.append(best_vloss)
            best_epoch.append(epoch)
            torch.save(model.state_dict(), root_data_path / 'net.pth')
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping")
                break

        epochs.append(epoch)
        # val_losses.append(val_loss.cpu().detach().numpy())
        # train_losses.append(train_loss.cpu().detach().numpy())
        val_losses.append(val_loss)
        train_losses.append(train_loss)

    train_finish = timer()

    print(f"Train time on {device}: {train_finish-train_start:.3f} seconds")

    # return epochs[1:], val_losses[1:], train_losses[1:]
    return epochs, val_losses, train_losses, loss_saved, best_epoch
