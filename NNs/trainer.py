from multiprocessing.connection import deliver_challenge
from tabnanny import process_tokens

from py import process
from methods.neural_network.FastNN import *
from pathlib import Path as path
from methods import *
#from methods.dataset import ChestHeartDataset
#from methods.visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
import methods.benchmarking
import cv2 as cv

torch.cuda.empty_cache()

N_EPOCHS = 20
BATCH_SIZE = 32
if __name__ == '__main__':
    # Dataset loader
    root_data_path = path.cwd() / 'data' / 'heart_augmented_COCO'

    training_ann = root_data_path / 'train.json'
    testing_ann = root_data_path / 'test.json'
    validate_ann = root_data_path / 'validation.json'

    train_heart_dataset = ChestHeartDataset(root_data_path, training_ann)
    test_heart_dataset = ChestHeartDataset(root_data_path, testing_ann)
    validate_heart_dataset = ChestHeartDataset(root_data_path, validate_ann)

    net = Network()

    criterion = nn.CrossEntropyLoss()

    training_data_loader = torch.utils.data.DataLoader(
        train_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testing_data_loader = torch.utils.data.DataLoader(
        test_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        validate_heart_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=0.1, momentum=0.9
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net = net.to(device)

    # Parameters for training

    train(net, N_EPOCHS, training_data_loader, validate_data_loader, device)
    torch.save(net.state_dict(), root_data_path / 'net.pth')
    net.load_state_dict(torch.load(root_data_path / 'net.pth'))
    net.eval()

    print("Test image")
    test_path = path.cwd() / 'test.png'
    test = cv.imread(test_path.as_posix())
    process_img = preprocess(test)
    print(process_img.shape)
    bbox_out = net(
        torch.from_numpy(
            process_img
            ).float()
            .unsqueeze(0)
            .to(device)
    )
    bbox_out = postprocess(bbox_out)
    process_img = cv.normalize(
        process_img, None, 
        0, 255, cv.NORM_MINMAX,
        cv.CV_8U
    )
    process_img = torch.from_numpy(
        process_img).to(device)
    bbox_out = torch.tensor(
        list(bbox_out)
    ).unsqueeze(0)
    print(process_img.shape)
    print(bbox_out.shape)
    print(bbox_out)

    result = torchvision.utils.draw_bounding_boxes(
        process_img,
        bbox_out,
        colors='red',
        width=5
    )
    show(result)
    plt.show()