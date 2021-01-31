#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:05:24 2020

@author: joshua
"""
import os
import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Data
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

G1 = torch.arange(0,NG1,1)          # Generating factor (Identity), {0,1,...,9}
G2 = torch.arange(0,NG2,1)           # Generating factor (Font color), {0,1,...,8}



fimages = os.path.join('..','data','t10k-images-idx3-ubyte')
flabels = os.path.join('..','data','t10k-labels-idx1-ubyte')

# Load images
with open(fimages, 'rb') as f:
    _, _, rows, cols = struct.unpack(">IIII", f.read(16))
    images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
images = np.tile(images[:, :, :, np.newaxis], 3)
# Load labels
with open(flabels, 'rb') as f:
    struct.unpack(">II", f.read(8))
    labels = np.fromfile(f, dtype=np.int8)
    labels = torch.from_numpy(labels.astype(np.int))

font_color = [[31, 119, 180],
              [255, 127, 14],
              [44, 160, 44],
              [214, 39, 40],
              [148, 103, 189],
              [140, 86, 75],
              [227, 119, 194],
              [127, 127, 127],
              [188, 189, 34],
              [12, 189, 34]]


validation_split = 0.1
batch_size = 64
                  
data_factor_list = []
data_onehot_list = []
data_X_list = []

for g1 in G1:
    mask = (labels==g1)
    print(g1)
    for g2 in G2:
        for g3 in G3:
            tmp_factor = np.array([g1,g2,g3])
            tmp_onehot = np.zeros(27)
            tmp_onehot[g1] = 1
            tmp_onehot[10+g2] = 1
            tmp_onehot[10+9+g3] = 1    
            
            tmp_images = copy.deepcopy(images[mask])
            tmp_images[:,:,:,0] = images[mask][:,:,:,0]/255*font_color[g2][0] + \
                                (255-images[mask][:,:,:,0])/255*back_color[g3][0]
            tmp_images[:,:,:,1] = images[mask][:,:,:,1]/255*font_color[g2][1] + \
                                (255-images[mask][:,:,:,1])/255*back_color[g3][1]
            tmp_images[:,:,:,2] = images[mask][:,:,:,2]/255*font_color[g2][2] + \
                                (255-images[mask][:,:,:,2])/255*back_color[g3][2] 

            
            num_samples = images[mask].shape[0]         # How many samples belongs to g1
            tmp_factor = np.tile(tmp_factor,(num_samples,1))
            tmp_onehot = np.tile(tmp_onehot,(num_samples,1))
            
            if g1+g2+g3 == 0:
                data_factor_list = tmp_factor
                data_X_list = tmp_images
                data_onehot_list = tmp_onehot
            else:
                data_X_list = np.concatenate([data_X_list,tmp_images],0)
                data_factor_list = np.concatenate([data_factor_list,tmp_factor],0)
                data_onehot_list = np.concatenate([data_onehot_list,tmp_onehot],0)

x = torch.tensor(data_X_list)                       # [N, 28,28,3]
y = torch.tensor(data_factor_list).int()             # [N, 3], i.e., the true generating factor
y_onehot = torch.tensor(data_onehot_list).int()        # [N, 27]
dataset=Data.TensorDataset(x, y, y_onehot)

np.random.seed(random_seed)
dataset_size = len(data_X_list)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)     
            
train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last = True)
validation_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last = True)






if __name__ == '__main__':
        
    big_images = np.zeros((280,280,3))
    for i in range(10):
        for j in range(10):
            rand = np.random.randint(0,720000)
            big_images[i*28:(i+1)*28,j*28:(j+1)*28,:] = data_X_list[rand]/255
    plt.imshow(big_images)   

