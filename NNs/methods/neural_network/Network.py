import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # CNNs for rgb images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)


        # Connecting CNN outputs with Fully Connected layers for classification
        #self.class_fc1 = nn.Linear(in_features=23232, out_features=240)
        #self.class_fc2 = nn.Linear(in_features=240, out_features=120)
        #self.class_out = nn.Linear(in_features=120, out_features=2)

        # Connecting CNN outputs with Fully Connected layers for bounding box
        self.box_fc1 = nn.Linear(in_features=23232, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_out = nn.Linear(in_features=120, out_features=4)


    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv5(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=4, stride=2)

        t = torch.flatten(t,start_dim=1)
        

        #class_t = self.class_fc1(t)
        #class_t = F.relu(class_t)

        #class_t = self.class_fc2(class_t)
        #class_t = F.relu(class_t)

        #class_t = F.softmax(self.class_out(class_t),dim=1)

        box_t = self.box_fc1(t)
        box_t = F.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)

        box_t = self.box_out(box_t)
        box_t = F.sigmoid(box_t)

        return box_t


def train(
    model, n_epochs, dataloader,
    valdataloader, device
):
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=0.9
    )
    epochs = []
    losses = []

    for epoch in range(n_epochs):
        tot_loss = 0
        train_start = time.time()
        model.train()
        for batch, (img, bbox, label) in enumerate(dataloader):
            img, bbox = img.to(device), bbox.to(device)

            optimizer.zero_grad()
            bbox_pred = model(img)

            box_loss = F.mse_loss(bbox_pred, bbox)
            box_loss.backward()

            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                    (time.time()-train_start)/60, end='\r')
        
        model.eval()
        for batch, (img, bbox, label) in enumerate(valdataloader):
            img, bbox = img.to(device), bbox.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                bbox_pred = model(img)

                box_loss = F.mse_loss(bbox_pred, bbox)
            
            tot_loss += box_loss.item()
            print("Test batch:", batch+1, " epoch: ", " ",
                    (time.time()-train_start)/60, end='\r')
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch", epoch, "Loss: ", tot_loss,
                "time: ", (time.time()-train_start)/60, " mins")