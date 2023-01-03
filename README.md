# Classification of fire and no fire images
The Dataset is obtained from kaggle flame detection competition: https://www.kaggle.com/code/truptisatsangi/fire-classification/data

Steps:
>> Download the data from kaggle link. The directory structure for data is given below
>> Directory Structure: data>train>Fire
                        data>train>NoFire
                        data>test>Fire
                        data>test>NoFire
                        
>> buildDataset.py: This module loads and transforms the data:  path = The path of train/ test directory; batch_size= desired batch size; 
                    resizeDimension = desired input size of the image is passed, Ex: (300,150) is the default dimension of image

>> cnnClassifier.py: Model is defined here.

>> dimenCheck.py: For the fully connected layer which if sequentially after Conv layer would require accurate output size of the Conv2D layer.
                  This routine helps to find the accurate output dimension without manually calculating.
                  
>> train.py: This is the main training loop

>> evaluate.py: This is the. testing loop
                    
>> Accuracy of the model for 10 epochs is 83% for 24 images
