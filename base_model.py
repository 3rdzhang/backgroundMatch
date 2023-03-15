from transformers import AutoImageProcessor, ViTModel
import torch
from PIL import Image
import  os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

 
from numpy import vstack
from pandas import read_csv

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# dataset definition
class CSVDataset(Dataset):
    def __init__(self, data_path):
        df = read_csv(data_path, header=None)
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])
 

# model definition
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(128, 16)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(16, 1)
        xavier_uniform_(self.hidden3.weight)
        self.hidden4 = Linear(2, 1)
        xavier_uniform_(self.hidden4.weight)
        self.act3 = Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        left = X[:,:768]
        right = X[:,768:]
        sim = torch.nn.functional.cosine_similarity(left, right)
        sim = torch.reshape(sim,(-1,1))
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = torch.cat([X, sim], -1)
        X = self.hidden4(X)
        X = self.act3(X)
        return X

# prepare the dataset
def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl
 
# train the model
def train_model(train_dl, model):
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

# prediction for data
def predict(ori_feat, back_feats, model):
    ret = []
    for back in back_feats:
        tmp = torch.cat([ori_feat[0], back[0]], -1)
        ret.append(tmp)
    pred_data = torch.stack(ret)
    print(pred_data.shape)
    yhat = model(pred_data)
    return yhat


def cal_score(model, img_feats, back_feats, topN = 3): 
    ori_img = list(img_feats.keys())[0]
    ori_feas = list(img_feats.values())[0]
    back = []
    key = []
    for k in back_feats:
        back.append(back_feats[k])
        key.append(k)
    y = predict(ori_feas, back, model)
    pred_result = {k:v.detach().numpy()[0] for k,v in zip(key, y)}
    ret = sorted(pred_result.items(), key=lambda d:d[1], reverse = True)
    return {ori_img : ret[:topN]}
