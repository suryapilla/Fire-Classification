
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms
from torchvision import datasets, transforms
from PIL import Image

import build_dataset
import classificationNetwork
import cnnClassifier

device='cpu'

input_size = 300*150
num_classes = 2
image_channels = 3
resizeDimension =  (300,150)

# Pass the path to test data
path_test = './data/test'
batch_size = 2
test_data = build_dataset.customDataLoad(path_test,batch_size,resizeDimension)

# model
model = cnnClassifier.convNet(num_classes).to(device)
model.load_state_dict(torch.load('./models/cnnModel_weights_10epochs.pth'))

## Method 1: Inference on test data
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_data:
        # images = images.unsqueeze(0)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the test images: {acc} %')


## Method 2: Inference on individual image

# test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=image_channels),transforms.Resize((300,150)),transforms.ToTensor()])

# image = Image.open('./data/test/Fire/Fire_745.png')
# image = test_transforms(image).float()
# image = image.unsqueeze(0) # used for CNNs
# # image = image.reshape(-1,input_size).to(device) # used for neural network

# ps = torch.exp(model.forward(image))
# _, predVal = torch.max(ps,1) 
# print(ps.float(),predVal)


## Method 3: Inference on a video

# import cv2
# import numpy as np
# webCam = cv2.VideoCapture(r'./fire.mp4')

# while(True):
#     re,frame = webCam.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
#     image = Image.fromarray(frame)
#     image = test_transforms(image).float()
#     # image = image.reshape(-1,input_size).to(device)
#     image = image.unsqueeze(0)
#     # dataiter = iter(image)
#     # images, labels = dataiter.next()
#     # images, labels = images.to(device), labels.to(device)
#     ps = torch.exp(model.forward(image))
#     _, predVal = torch.max(ps,1) 
#     txt = str(predVal.numpy())
#     cv2.putText(frame, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color = (255,0,255), thickness = 2)
#     cv2.imshow("Win",frame)
#     if(cv2.waitKey(1)==ord('q')):
#         break
#     # print(predVal)
# cv2.destroyAllWindows()
# webCam.release()
