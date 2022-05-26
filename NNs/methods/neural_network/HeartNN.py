from tracemalloc import start
import torch
import torch.nn as nn
import torchvision


class HeartNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.MaxPool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)

        self.fc1 = nn.Linear(512 * 26 * 26, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.MaxPool(self.relu(self.conv2(x)))

        x = self.relu(self.conv3(x))
        x = self.MaxPool(self.relu(self.conv4(x)))

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.MaxPool(self.relu(self.conv6(x)))

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.MaxPool(self.relu(self.conv8(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        return x


def loss_fn(pred, label):
    loss = torchvision.ops.box_iou(pred, label)

    loss = 1 - loss

    return loss.mean()


def train_nn_one_epoch(net, train_loader, optimizer, device):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        print(type(data))
        print(len(data))
        imgs, boxes, labels = data

        imgs = imgs.to(device)
        boxes = boxes.to(device)

        optimizer.zero_grad()

        outputs = net(imgs)

        loss = loss_fn(outputs, boxes)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            print('batch {} loss: {}'.format(i+1, last_loss))
            running_loss = 0

    return last_loss


def train_Heartnn(
    model, n_epochs,
    train_loader, eval_loader,
    optimizer, device
):
    for epoch in range(n_epochs):
        model.train(True)
        avg_loss = train_nn_one_epoch(
            model, train_loader,
            optimizer, device
        )
        model.train(False)

