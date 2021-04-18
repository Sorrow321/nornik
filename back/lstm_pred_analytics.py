#lstm_pred_analytics.py
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch
from torch import nn
import pandas as pd
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


class CNN_with_torch_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN_with_torch_lstm, self).__init__()
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Sequential(nn.Linear(hidden_dim*30, 256), nn.ReLU()) #30 - time periods
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, 2, 1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(4, 2, 1),
                                       nn.ReLU())
        self.linear_out = nn.Sequential(nn.Linear(64 * 2 * 2, output_dim),nn.Sigmoid())#nn.Linear(64 * 2 * 2, output_dim)
 
    def forward(self, X, hidden):
        
        output_lstm, _ = self.lstm(X)
        output_lstm = output_lstm.view(X.shape[0], -1)
        output1 = self.linear(output_lstm)
        output1 = output1.view(X.shape[0], 1, 16, 16)
        output_cnn = self.conv_layer(output1)
        output_cnn = output_cnn.view(X.shape[0], -1)
        output = self.linear_out(output_cnn)
 
        return output

def _epoch(network, loss, loader, optimizer, device='cpu', p = 'train'):
    losses = []
    for X, y in loader:
        X = X.type(dtype=torch.FloatTensor)
        y = y.type(dtype=torch.FloatTensor)
        hidden = (torch.zeros(X.shape[1], network.hidden_size).type(dtype=torch.FloatTensor).to(device),
                        torch.zeros(X.shape[1], network.hidden_size).type(dtype=torch.FloatTensor).to(device))
        #hidden = 100
        X = X.to(device)
        y = y.to(device)
        prediction = network(X, hidden)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.item())
        if p == 'train':
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
    return losses

def train(network, train_loader, val_loader, epochs, learning_rate, device='cpu'):
    phase = ['train', 'val']
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss = nn.MSELoss()
    train_loss_epochs = []
    network = network.to(device)
    network = network.train()
    try:
        for epoch in range(epochs):
            for p in phase:
                if p == 'train':
                    losses = _epoch(network, loss, train_loader, optimizer, device, p)
                else:
                    losses = _epoch(network, loss, val_loader, optimizer, device, p)
                    train_loss_epochs.append(np.mean(losses))
                print(epoch)
    except KeyboardInterrupt:
        if tolerate_keyboard_interrupt:
            pass
        else:
            raise KeyboardInterrupt
    torch.save(network.state_dict(), 'PA.pth')
    return train_loss_epochs