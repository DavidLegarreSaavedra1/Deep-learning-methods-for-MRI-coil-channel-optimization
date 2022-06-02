from ctypes import WinDLL
from logging import root
from .Loss import *
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
    loss = nn.SmoothL1Loss()(pred, target)
    #loss = nn.MultiLabelSoftMarginLoss()(pred, target).sum()
    #loss = IoU_loss(pred,target)
    return loss

def train(
    model, n_epochs, dataloader,
    valdataloader, device, root_data_path
):

    Adam = True
    learning_rate = 0.01
    min_val_loss = 1_000_000
    n_epochs_stop = 7
    wd = 0
    n_no_improve = 0
    early_stop = False


    if Adam:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=0.9, weight_decay=wd 
        )

    epochs = []
    losses = []
    train_losses = []

    for epoch in range(n_epochs):

        tot_loss = 0
        train_loss = 0
        train_start = time.time()
        model.train()

        for batch, (img, bbox, label) in enumerate(dataloader):

            img, bbox = img.to(device), bbox.to(device)
            label = label.to(device)
            img = img.float()
            bbox = bbox.float()


            bbox_pred = model(img)
            bb_loss = loss_fn(bbox_pred, bbox)

            optimizer.zero_grad()
            bb_loss.backward()
            train_loss += bb_loss.item()
            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                    (time.time()-train_start)/60, end='\r')
        
        model.eval()
        for batch, (img, bbox, label) in enumerate(valdataloader):
            img, bbox = img.to(device), bbox.to(device)
            label = label.to(device)
            img = img.float()
            bbox = bbox.float()

            optimizer.zero_grad()
            with torch.no_grad():
                bbox_pred = model(img)

                box_loss = loss_fn(bbox_pred, bbox)

            tot_loss += box_loss.item()
            print("Val batch:", batch+1, " epoch: ", " ",
                    (time.time()-train_start)/60, end='\r')
            torch.cuda.empty_cache()
            
        epochs.append(epoch)
        losses.append(tot_loss)
        train_losses.append(train_loss)
        print("Epoch", epoch, "Loss: ", tot_loss, 
                "time: ", (time.time()-train_start)/60, " mins")

        if tot_loss < min_val_loss:
            n_no_improve = 0
            min_val_loss = tot_loss
            torch.save(model.state_dict(), root_data_path / 'net.pth')
        else:
            n_no_improve += 1
        
        if epoch > 5 and n_no_improve == n_epochs_stop:
            print("Early stopped")
            early_stop = True
            break

    return epochs, losses, train_losses