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
        # Input 1 * 512 * 512 B&W
        self.conv1 = nn.Conv2d(1, 6, 5) # 6 * 254 * 254
        self.conv2 = nn.Conv2d(6, 16, 5) # 16 * 125 * 125
        self.conv3 = nn.Conv2d(16, 32, 5) # 32 * 60 * 60
        self.conv4 = nn.Conv2d(32, 64, 5) # 64 * 28 * 28
        self.conv5 = nn.Conv2d(64, 192, 5) # 192 * 12 * 12
        
        # Label classification
        self.class_fc1 = nn.Linear(192 * 5 * 5, 240)
        self.class_fc2 = nn.Linear(240, 120)
        self.class_ = nn.Linear(120, 2)

        # BBox regression
        self.fc1 = nn.Linear(192 * 5 * 5 , 240)
        self.fc2 = nn.Linear(240, 120)
        #self.fc3 = nn.Linear(120, 32)
        self.bb = nn.Linear(120, 4) 

        self.pool = nn.MaxPool2d(2,2)

        self.loss = nn.L1Loss()
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = F.avg_pool2d(x, kernel_size=4, stride=2)
        x = torch.flatten(x, 1)

        classx = F.relu(self.class_fc1(x))
        classx = F.relu(self.class_fc2(classx))
        classx = F.softmax(self.class_(classx), dim=1)

        bbx = F.relu(self.fc1(x))
        bbx = F.relu(self.fc2(bbx))
        bbx = torch.sigmoid(self.bb(bbx))

        return classx, bbx



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
    for epoch in range(n_epochs):
        #net.train()
        running_loss = 0
        start = time.time()
        for batch, (img, bbox, label) in enumerate(train_data_loader):
            #print(img)
            #img = img.cuda().float()
            #bbox = bbox.cuda().float() 
            img = img.to(device)
            bbox = bbox.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            label_pred, out_bb = net(img)

            label_pred = torch.max(label_pred, 1)[0]

            #loss_bb = torch.cdist(out_bb, bbox)
            loss_bb = IoU_loss(out_bb, bbox)
            label_loss = F.l1_loss(label_pred, label)
            #loss_bb = IoU_loss(out_bb, bbox)
            #loss_bb = loss_bb.sum()
            loss = (label_loss+loss_bb)
            loss.backward()
            optimizer.step()
             # Gather data and report
            running_loss += loss.item()
            if batch % 10 == 9:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(batch + 1, last_loss))
                running_loss = 0.


def data_testing(model, test_loader):
    device='cuda'
    dataiter = iter(test_loader)
    images, bbox, label = dataiter.next()

    images, bbox, label = images.to(device), bbox.to(device), label.to(device)[:,0]

    label_pred, bbox_pred = model(images)

    _, predicted = torch.max(label_pred, 1)

    print(type(_))
    
    print(_.shape)

    print(type(predicted))
    print(predicted.shape)



