# Libraries
import os
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from data import prepared_data
from Transformer import VitTransformer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
img_size = 224
img_channels = 3
batch_size = 8
batch_size = 35
latent_size = 768
dropout = 0.1
num_encoders = 12
num_classes = 4
learning_rate = 10e-3
weight_decay = 0.03
epochs = 50


# Data preparation 
# The loader
train_dir = 'my_custom_data/Training'
train_loader = prepared_data(train_dir,img_size,img_channels,batch_size)


# Training process
model = VitTransformer(latent_size,num_encoders,num_classes,dropout,device).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,weight_decay = weight_decay)
loss = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

model.train()
for epoch in tqdm(range(epochs),total = epochs):
    total_loss = 0.0
    
    for idx,(imgs,labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(imgs)
        loss_ = loss(output,labels)
        loss_.backward()
        optimizer.step()
        
        total_loss += loss_.item()
        
        if (idx + 1) % 40 == 0:
            print(f'Epoch {epoch + 1}/{epochs} Batch {idx + 1}, loss = {total_loss / 40:.3f}')
            total_loss = 0.0
    lr_scheduler.step()