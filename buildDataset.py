import torch
import torchvision
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# path = "./data/train"
# batch_size = 100
# resize image to 300x150:=>  resizeDimension=(300,150)

def customDataLoad(path,batch_size,resizeDimension):
    
    # define transformation to input images
    # original image is resized to 300x150
    # Original image is converted to grayscale

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(resizeDimension),transforms.ToTensor()])

    # Apply transform and import dataset
    dataset = datasets.ImageFolder(path,transform=transform)
    
    # Load data into dataloader with specific batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

