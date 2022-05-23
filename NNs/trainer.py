from methods.neural_network.FastNN import *
from pathlib import Path as path
from methods.dataset import ChestHeartDataset
from methods.visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import methods.benchmarking




if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented_COCO'
    print(path.cwd())
    training_ann = root_data_path / 'train.json'
    testing_ann = root_data_path / 'test.json'

    train_heart_dataset = ChestHeartDataset(root_data_path, training_ann)
    test_heart_dataset = ChestHeartDataset(root_data_path, testing_ann)

    net = FastNN().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    batch_size = 32
    training_data_loader = torch.utils.data.DataLoader(
        train_heart_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    testing_data_loader = torch.utils.data.DataLoader(
        test_heart_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    img, bbox = next(iter(training_data_loader))

    # Parameters for training
    epochs = 4
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    
    optimizer = torch.optim.Adam(
        parameters, lr=0.005
    )

   # train_nn(net, epochs, training_data_loader, optimizer)
   # torch.save(net.state_dict(), root_data_path / 'net.pth') 
    net.load_state_dict(torch.load(root_data_path / 'net.pth'))
    
    data_testing(net, testing_data_loader)
