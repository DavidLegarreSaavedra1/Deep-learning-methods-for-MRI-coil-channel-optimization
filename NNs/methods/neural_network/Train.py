from ctypes import WinDLL
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from logging import root
from datetime import datetime
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
    return loss


def train_(
    model, n_epochs, dataloader,
    valdataloader, device, root_data_path,
    img_size
):

    Adam = True
    learning_rate = 0.01
    min_val_loss = 1_000_000
    n_epochs_stop = 7
    wd = 0.5
    n_no_improve = 0
    early_stop = False

    transform = T.Resize(img_size)

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
    original_sizes = []

    for epoch in range(n_epochs):

        tot_loss = 0
        train_loss = 0
        train_start = time.time()
        model.train()

        for batch, (img, bbox) in enumerate(dataloader):

            for i in img:
                sizes = i.shape[-2:]
                original_sizes.append(sizes)

            img = transform(img)
            img, bbox = img.to(device), bbox.to(device)
            img = img.float()
            bbox = bbox.float()

            bbox_pred = model(img)

            bbox_ = []
            bbox_pred = bbox_pred*512

            bb_loss = loss_fn(bbox_pred, bbox)

            optimizer.zero_grad()
            bb_loss.backward()
            train_loss += bb_loss.item()
            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-train_start)/60, end='\r')

        model.eval()
        for batch, (img, bbox) in enumerate(valdataloader):
            img = transform(img)
            img, bbox = img.to(device), bbox.to(device)
            img = img.float()
            bbox = bbox.float()

            optimizer.zero_grad()
            with torch.no_grad():
                bbox_pred = model(img)
                bbox_pred = bbox_pred*512
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


def train_one_epoch(
    model, epoch_index, tb_writer,
    training_loader, optimizer,
    device, resizer
):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = resizer(inputs)

        # Zero your gradients for every batch!
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 2 == 1:
            last_loss = running_loss / 2  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss), end='\r')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(
    model, EPOCHS, train_loader,
    validation_loader, device, root_data_path,
    img_size
):

    Adam = True
    learning_rate = 0.01
    wd = 0.5

    transform = T.Resize(img_size)

    if Adam:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=0.9, weight_decay=wd
        )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.
    epochs = []
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train()
        avg_loss = train_one_epoch(
            model, epoch_number, writer,
            train_loader, optimizer,
            device, transform
        )

        torch.cuda.empty_cache()
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = transform(vinputs)
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), root_data_path / 'net.pth')

        epoch_number += 1
        epochs.append(epoch)
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)
    return epochs, val_losses, train_losses

