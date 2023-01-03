# This routine is used to check the dimension of the input images so that inputs for the fully connected layer is given accordingly

import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

import build_dataset

input_size = 150*300
device = 'cpu'


path = r"./data/test"
batch_size =2
resizeDimension = (300,150)
train_data = build_dataset.customDataLoad(path,batch_size,resizeDimension)
dataiter = iter(train_data)
images, labels = dataiter.next()

conv1 = nn.Conv2d(1,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
print(images.shape)
x = conv1(images)
print(x.shape)

x=pool(x)
print(x.shape)

x=conv2(x)
print(x.shape)

x=pool(x)
print(x.shape)