from matplotlib import image
from numpy.core.numeric import _tensordot_dispatcher
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.optim as optim

import matplotlib.pyplot as plt
import time
import copy
import os
import cv2
from PIL import Image

data_transforms = transforms.Compose([
        transforms.Resize((200, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class FramesDataset(Dataset):
    def __init__(self, root_dir, transforms, length):
        self.images = []
        self.target = []
        for folder in Path(root_dir).iterdir():
            print(folder)
            self.images += [transforms(Image.open(str(file))) for file in (folder / 'images').iterdir()]
            with open((folder / 'labels.txt'), 'rt') as file:
                lines = np.array([int(x) for x in file.readline().replace(',', ' ').strip().split()])
                target = np.zeros(len(self.images))
                target[lines] = 1
                self.target += list(target)
        self.images = torch.stack(self.images)
        self.target = np.array(self.target)
        self.length = length

    def __len__(self):
        return len(self.images) - self.length + 1

    def __getitem__(self, index):
        return (self.images[index : index + self.length], self.target[index : index + self.length])

class CLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(CLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = models.resnet18(pretrained=True)
        for name, param in self.feature_extractor.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            pass
            #param.requires_grad = False
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, frames):
        frames_embeddings = self.feature_extractor(frames.view(frames.shape[0] * frames.shape[1], frames.shape[2], frames.shape[3], frames.shape[4]))
        frames_embeddings = frames_embeddings.reshape((frames.shape[0], frames.shape[1], -1))
        lstm_out, _ = self.lstm(frames_embeddings)
        tag_space = self.hidden2tag(lstm_out.view(frames.shape[0] * frames.shape[1], -1))
        return tag_space

batch_size = 128
seq_len = 10
n_epochs = 50

train_dataset = FramesDataset(root_dir='input/', transforms=data_transforms, length=5)
test_dataset = FramesDataset(root_dir='test_input/', transforms=data_transforms, length=5)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('Dataset size: ', len(train_dataset))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CLSTM(512, 64, 1)
model.to(device)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    
    return acc


def test(net):
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.view(labels.shape[0] * labels.shape[1], -1)
            labels = labels.to(device)

            outputs = net(inputs)
            proba = torch.sigmoid(outputs)
            predict = torch.round(proba)
            print('-----------------')
            print(predict.sum())
            print('acc: ', (predict == labels).float().sum() / labels.shape[0])
            print('------------------')
            break

def train(net):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print('Epoch number ', epoch)

        epoch_acc = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.view(labels.shape[0] * labels.shape[1], -1)
            labels = labels.to(torch.float32).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            epoch_acc += acc.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.item()
            #if i % 128 == 0:    # print every 2000 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #        (epoch + 1, i + 1, running_loss / 2000))
            #    running_loss = 0.0
        
        print(f'Epoch {epoch} | Loss {epoch_loss / (i+1)} | Acc {epoch_acc / (i+1)}')
            
        if epoch % 5 == 0:
            test(model)
            torch.save(model.state_dict(), f'clstm_hid64/base{epoch}.pth')
    print('Finished Training')

train(model)

torch.save(model.state_dict(), 'clstm_hid64/base.pth')



test(model)