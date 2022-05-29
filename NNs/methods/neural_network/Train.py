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
    #loss = nn.CrossEntropyLoss()(pred, target)
    #loss = nn.MultiLabelSoftMarginLoss()(pred, target).sum()
    loss = IoU_loss(pred,target)
    return loss

def train(
    model, n_epochs, dataloader,
    valdataloader, device
):
    optimizer = optim.SGD(
        model.parameters(), lr=1e-5,
        momentum=0.9
    )
    optimizer = optim.Adamax(
        model.parameters(), lr=1e-2
    )
    epochs = []
    losses = []

    for epoch in range(n_epochs):
        tot_loss = 0
        train_start = time.time()
        model.train()
        for batch, (img, bbox, label) in enumerate(dataloader):
            img, bbox = img.to(device), bbox.to(device)
            label = label.to(device)


            class_pred, bbox_pred = model(img)
            bb_loss = loss_fn(bbox_pred, bbox)
            bb_loss = bb_loss.sum()
            class_loss = F.cross_entropy(class_pred, label, reduction='sum')

            loss = class_loss + bb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                    (time.time()-train_start)/60, end='\r')
        
        model.eval()
        for batch, (img, bbox, label) in enumerate(valdataloader):
            img, bbox = img.to(device), bbox.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                class_pred, bbox_pred = model(img)

                class_loss = F.cross_entropy(class_pred, label, reduction='sum')
                box_loss = loss_fn(bbox_pred, bbox).sum()

                loss = class_loss + box_loss
            
            tot_loss += loss.item()
            print("Test batch:", batch+1, " epoch: ", " ",
                    (time.time()-train_start)/60, end='\r')
            torch.cuda.empty_cache()
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch", epoch, "Loss: ", tot_loss, 
                "time: ", (time.time()-train_start)/60, " mins")

    return epochs, losses