from nn import *
from dataset import *
from pathlib import Path as path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented'
    print(path.cwd())
    training_ann = root_data_path / 'train.json'

    heart_dataset = ChestHeartDataset(root_data_path, training_ann)

    # Network

    net = HeartDetectorNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    data_loader = torch.utils.data.DataLoader(
        heart_dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net.to(device)
    n_epochs = 10

    for epoch in range(2):
        net.train()
        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in b.items()}
                           for b in annotations]

            # zero the gradients
            optimizer.zero_grad()

            # forward
            output = net(imgs)
            loss = criterion(output, annotations)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
