import torch
import torch.nn as nn
import torch.nn.functional as F

image_channels = 1
class convNet(nn.Module):
    def __init__(self, num_classes):
        super(convNet,self).__init__()
        self.conv1 = nn.Conv2d(image_channels,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        # this input dimension needs to be calculated based on the layers above
        # A python file "dimenCheck.py" gives out the dimension of the output of conv layers
        self.fc1 = nn.Linear(16*72*34, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*72*34)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



