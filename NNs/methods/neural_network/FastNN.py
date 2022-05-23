import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastNN(nn.Module):
    # Input images of size 512 512
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.bb = nn.Linear(32, 4) 
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.bb(x)

        return x

def train_nn(net, epochs, 
        data_loader, optimizer):
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
            
            loss_bb = F.smooth_l1_loss(out_bb, bbox)
            loss_bb = loss_bb.sum()
            optimizer.zero_grad()
            loss_bb.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss_bb.item()
        train_loss = sum_loss/total
        print(f"Train_loss: {train_loss}")


def data_testing(model, test_loader):
    dataiter = iter(test_loader)
    images, bboxes = dataiter.next()

    images = images.cuda().float()
    bboxes = bboxes.cuda().float()

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print(type(_))
    
    print(_.shape)

    print(type(predicted))
    print(predicted.shape)



