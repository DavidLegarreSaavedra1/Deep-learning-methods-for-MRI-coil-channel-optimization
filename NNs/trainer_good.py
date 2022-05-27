import torch
from methods import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path as path


BATCH_SIZE = 4

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_data_path = path.cwd() / 'data' / 'heart_augmented_COCO'

    training_ann = root_data_path / 'train.json'
    testing_ann = root_data_path / 'test.json'
    validate_ann = root_data_path / 'validation.json'

    train_heart_dataset = ChestHeartDataset(root_data_path, training_ann)
    test_heart_dataset = ChestHeartDataset(root_data_path, testing_ann)
    validate_heart_dataset = ChestHeartDataset(root_data_path, validate_ann)
    training_data_loader = torch.utils.data.DataLoader(
        train_heart_dataset,
        batch_size=BATCH_SIZE,
        num_workers = 4,
        shuffle=True,
    )
    testing_data_loader = torch.utils.data.DataLoader(
        test_heart_dataset,
        batch_size=BATCH_SIZE,
        num_workers = 4,
        shuffle=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        validate_heart_dataset,
        batch_size=BATCH_SIZE,
        num_workers = 4,
        shuffle=True,
    )
    dataloaders = {
        "train" : training_data_loader,
        "test" : testing_data_loader,
        "val" : validate_data_loader
    }

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(device)
    # Get a batch of training data
    inputs, classes, _ = next(iter(train_heart_dataset))

    print(inputs.shape)
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out,title='test')



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25, dataloaders=dataloaders, device=device)

    visualize_model(model)
