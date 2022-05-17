from FastNN import Cifar10CnnModel, FastNN
from nn import *
from dataset import *
from pathlib import Path as path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F



if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented_COCO'
    print(path.cwd())
    training_ann = root_data_path / 'train.json'

    heart_dataset = ChestHeartDataset(root_data_path, training_ann)

    # Network

    net = FastNN().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    batch_size = 64
    data_loader = torch.utils.data.DataLoader(
        heart_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    # Parameters for training
    epochs = 2
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=0.006
    )

    # Training
    idx = 0
    for i in range(epochs):
        net.train()
        total = 0
        sum_loss = 0
        for img, bbox in data_loader:
            #print(img)
            batch = len(img)
            img = img.cuda().float()
            bbox = bbox.cuda().float() 
            out_bb = net(img)
            
            loss_bb = F.l1_loss(out_bb, bbox)
            loss_bb = loss_bb.sum()
            optimizer.zero_grad()
            loss_bb.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss_bb.item()
        train_loss = sum_loss/total
        print(f"Train_loss: {train_loss}")
