# Libraries
import os
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# Data inspection
def prepared_data(train_dir,img_size,img_channels,batch_size):
    train_cls = os.listdir(train_dir)
    print(f'No.of training classes = {len(train_cls)}')
    
    for i in range(len(train_cls)):
        train_imgs = os.listdir(os.path.join(train_dir,train_cls[i]))
        print(f'No of images in training set for {train_cls[i]} = {len(train_imgs)}')

    train_dir = 'data/Training'

    transform = transforms.Compose(
                                    [
                                        transforms.Resize((img_size,img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean = [0.5 for _ in range(img_channels)],
                                            std = [0.5 for _ in range(img_channels)]
                                                            ) 
                                    ]
                                )

    train_data = torchvision.datasets.ImageFolder(root = train_dir,transform = transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size = batch_size,shuffle = True)
    
    train_cls = train_data.class_to_idx
    print(f'Training classes = {train_cls}')

    classes = {0:'glioma_tumor',1:'meningioma_tumor',2:'no_tumor',3:'pituitary_tumor'}
    plt.figure(figsize = (15,8))
    imgs,labels = next(iter(train_loader))

    for i,img in enumerate(imgs):
        plt.subplot(5,7,i + 1)
        img = img.permute(1,2,0)
        img = img * torch.tensor([0.5 for _ in range(img_channels)]) + torch.tensor([0.5 for _ in range(img_channels)])
        plt.imshow(img)
        plt.title(classes[labels[i].item()])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return train_loader