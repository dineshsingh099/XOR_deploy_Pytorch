import torch 
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset,DataLoader 
import pandas as pd
import numpy as np

import os
import pickle
from src import datasets

from src.config import config as c
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

class xor_dataset(Dataset):
    def __init__(self, data):
        self.training_data = data

    def __len__(self):
        return len(self.training_data)
        
    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        X_train = torch.tensor(row.iloc[0:2].values, dtype=torch.float32)
        Y_train = torch.tensor(row.iloc[2], dtype=torch.float32)
        return X_train, Y_train

def initialize_data():
    data = load_dataset("train.csv")
    c.X_train = data.iloc[:, :-1].values 
    c.Y_train = data.iloc[:, -1].values.reshape(-1, 1)
    c.training_data = data

initialize_data()
    
Xordataset = xor_dataset(c.training_data)
data_gen = DataLoader(dataset= Xordataset, batch_size= c.mb_size)

# Functional API
class functional_mlp(nn.Module):

    def __init__(self):
        super().__init__()

        self.first_hidden_layer = nn.Linear(in_features=2, out_features=4, device="cpu")
        self.second_hidden_layer = nn.Linear(in_features=4, out_features=2, device="cpu")
        self.output_layer = nn.Linear(in_features=2, out_features=1, device="cpu")

    def forward(self,inp):

        first_hidden_layer_out = F.relu(self.first_hidden_layer(inp))
        second_hidden_layer_out = F.relu(self.second_hidden_layer(first_hidden_layer_out))
        nn_out = F.sigmoid(self.output_layer(second_hidden_layer_out))

        return nn_out
    
functional_nn = functional_mlp()
bce_loss = nn.BCELoss()
optimizer = torch.optim.RMSprop(functional_nn.parameters(), lr=0.01)

for e in range(c.epochs):
    running_loss = 0.0
    for X_train_mb, Y_train_mb in data_gen:

        optimizer.zero_grad()
       
        nn_out = functional_nn(X_train_mb)
        nn_out = nn_out.view(-1)
        loss_func = bce_loss(nn_out,Y_train_mb)
        loss_func.backward()
        
        optimizer.step()

        running_loss += loss_func.item()

    avg_loss = running_loss / len(data_gen)

    print("Epoch # {}, loss function value {:.6f}".format(e+1, avg_loss))

if __name__ == "__main__":
    save_model(functional_nn, c)