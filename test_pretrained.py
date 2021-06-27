#imports
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torchvision
import pathlib
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2

data_path = '/home/fri/python_cnn_test'
videos = glob.glob(data_path + "/*.mp4")
print(videos)
features = None
labels = None
label = -1
for video in videos:
  label = label + 1
  vid = cv2.VideoCapture(video)
  fps = int(vid.get(cv2.CAP_PROP_FPS))
  frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f"FPS: {fps} Frame Width: {frame_width} Frame Height: {frame_height}")
  length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
  feat_init = 0 if features is None else features.shape[0]
  for frame_index in range(10):
    processing, frame = vid.read()
    frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1], frame.shape[2]))
    features = frame if features is None else np.concatenate((features, frame))
  temp = np.full((features.shape[0] - feat_init), label)
  labels = temp if labels is None else np.concatenate((labels, temp))
print(features.shape)
print(labels.shape)

i = 0
plt.figure(figsize = (10,10))
plt.subplot(221), plt.imshow(features[0], cmap="brg")
plt.subplot(222), plt.imshow(features[5], cmap="brg")
plt.subplot(223), plt.imshow(features[10], cmap="brg")
plt.subplot(224), plt.imshow(features[15], cmap="brg")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=2, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(int(1920 * 1080 / 4), 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
features = np.reshape(features, (features.shape[0], features.shape[3], features.shape[1], features.shape[2]))
features = features.astype(np.float32)
features = torch.from_numpy(features)
labels = labels.astype(int)
labels = torch.from_numpy(labels)
print(features.shape)
print(labels.shape)
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size = 0.1)
model = ConvNet()
# # defining the optimizer
# optimizer = Adam(model.parameters(), lr=0.001)
# # defining the loss function
# criterion = CrossEntropyLoss()
# # checking if GPU is available
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
    
# print(model)
# def train(epoch):
#     model.train()
#     tr_loss = 0
#     # getting the training set
#     x_train, y_train = Variable(train_x), Variable(train_y)
#     # getting the validation set
#     x_val, y_val = Variable(val_x), Variable(val_y)
#     # converting the data into GPU format
#     if torch.cuda.is_available():
#         x_train = x_train.cuda()
#         y_train = y_train.cuda()
#         x_val = x_val.cuda()
#         y_val = y_val.cuda()

#     # clearing the Gradients of the model parameters
#     optimizer.zero_grad()
    
#     # prediction for training and validation set
#     output_train = model(x_train)
#     output_val = model(x_val)

#     # computing the training and validation loss
#     loss_train = criterion(output_train, y_train)
#     loss_val = criterion(output_val, y_val)
#     train_losses.append(loss_train)
#     val_losses.append(loss_val)

#     # computing the updated weights of all the model parameters
#     loss_train.backward()
#     optimizer.step()
#     tr_loss = loss_train.item()
#     if epoch%2 == 0:
#         # printing the validation loss
#         print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

# # defining the number of epochs
# n_epochs = 5
# # empty list to store training losses
# train_losses = []
# # empty list to store validation losses
# val_losses = []
# # training the model
# for epoch in range(n_epochs):
#     train(epoch)

model.load_state_dict(torch.load("model_weights.pth"))

with torch.no_grad():
    output = model(train_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
print(accuracy_score(train_y, predictions))
