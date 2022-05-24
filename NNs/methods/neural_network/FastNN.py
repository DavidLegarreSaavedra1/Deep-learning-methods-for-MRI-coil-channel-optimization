import enum
from matplotlib.pyplot import box
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

class FastNN(nn.Module):
    # Input images of size 512 512
    def __init__(self):
        super(FastNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.conv5 = nn.Conv2d(64, 192, 5)
        self.fc1 = nn.Linear(192 * 27 * 27 , 240)
        self.fc2 = nn.Linear(240, 120)
        #self.fc3 = nn.Linear(120, 32)
        self.bb = nn.Linear(120, 4) 
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = torch.sigmoid(self.bb(x))

        return x



def IoU_loss(predict_bbox, target_bbox, smooth=1e-6):

    loss = torchvision.ops.box_iou(predict_bbox, target_bbox)

    #loss = torch.clamp(loss, min=-1, max=1)
    loss = 1 - loss
    
    return loss.mean()


def train_nn(net, n_epochs, 
        train_data_loader, eval_data_loader,
        optimizer, device):
    # Training
    idx = 0
    total_loss = 0
    epochs = []
    losses = []
    for epoch in range(n_epochs):
        net.train()
        start = time.time()
        for batch, (img, bbox) in enumerate(train_data_loader):
            #print(img)
            batch = len(img)
            #img = img.cuda().float()
            #bbox = bbox.cuda().float() 
            img, bbox = img.to(device), bbox.to(device)
            optimizer.zero_grad()
            out_bb = net(img)

            #loss_bb = F.smooth_l1_loss(out_bb, bbox)
            #loss_bb = F.mse_loss(out_bb, bbox)
            loss_bb = IoU_loss(out_bb, bbox)
            #loss_bb = loss_bb.sum()
            loss_bb.backward()
            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-start)/60, end='\r')

        net.eval()
        for batch, (img, bbox) in enumerate(eval_data_loader):
            img, bbox = img.to(device), bbox.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                out_bb = net(img)
                loss_bb = IoU_loss(out_bb, bbox)
            total_loss += loss_bb.item()
            print("Test batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-start)/60, end='\r')

        epochs.append(epoch)
        losses.append(total_loss)
        print("Epoch", epoch, "loss:",
              total_loss, " time: ", (time.time()-start)/60, " mins")

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



