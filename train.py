import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms

import build_dataset
import classificationNetwork
import cnnClassifier

input_size = 300*150
hidden_size = 800
num_classes = 2
num_epochs = 10
batch_size = 100
resizeDimension = (300,150)
lr = 0.001

# Pass the path to train data
path_train = r"./data/train"
# Load train data
train_data = build_dataset.customDataLoad(path_train,batch_size,resizeDimension)

device='cpu'
model = cnnClassifier.convNet(num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

n_total_step = len(train_data)

# Training loop
for epochs in range(num_epochs):
    for i, (images,labels) in enumerate(train_data):
        # images = images.reshape(-1,input_size).to(device)
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1)%100==0:
        print(f'epochs {epochs+1} / {num_epochs},step {i+1}/{n_total_step}, loss = {loss.item():.3f}')

torch.save(model.state_dict(),'./models/cnnModel_weights_10epochs.pth')

