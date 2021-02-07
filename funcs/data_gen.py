#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:08:22 2020
For the given generating factors, we consider three different kinds of data:
    1. Linear map to high-dim
    2. Non-linear map to high-dim
    3. Map to visual signal

@author: joshua
"""
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data 
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import copy
from PIL import Image
import matplotlib.pyplot as plt

from .configs import *
from .utils import *

G1 = torch.arange(0,NG1,1)          # Generating factor (Identity), {0,1,...,9}
G2 = torch.arange(0,NG2,1)           # Generating factor (Font color), {0,1,...,9}


def _generate_Color_MNIST(G_MAPPING, batch_size=64,random_seed=42, path='data'):
    random_seed = random_seed
    
    fimages = os.path.join(path,'t10k-images-idx3-ubyte')
    flabels = os.path.join(path,'t10k-labels-idx1-ubyte')
    
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
             
    data_x_list = []
    data_y_list = []
    data_f_list = []
    zs_x_list = []
    zs_y_list = []
    zs_f_list = []    
         
    
    for g1 in G1:
        mask = (labels==g1)
        for g2 in G2:
            tmp_factor = np.array([g1,g2])
            tmp_onehot = np.zeros(NG1+NG2)
            tmp_onehot[g1] = 1
            tmp_onehot[NG1+g2] = 1 
        
            tmp_images = copy.deepcopy(images[mask])
            tmp_images[:,:,:,0] = images[mask][:,:,:,0]/255*font_color[g2][0] 
            tmp_images[:,:,:,1] = images[mask][:,:,:,1]/255*font_color[g2][1]
            tmp_images[:,:,:,2] = images[mask][:,:,:,2]/255*font_color[g2][2]
        
            num_samples = images[mask].shape[0]         # How many samples belongs to g1
            tmp_factor = np.tile(tmp_factor,(num_samples,1))
            tmp_onehot = np.tile(tmp_onehot,(num_samples,1))
            tmp_y = G_MAPPING(tmp_onehot).detach()
            if (g1,g2) not in ZS_TABLE:
                if len(data_x_list)==0:
                    data_x_list = tmp_images
                    data_y_list = tmp_y
                    data_f_list = tmp_factor
                else:
                    data_x_list = np.concatenate([data_x_list,tmp_images],0)
                    data_y_list = np.concatenate([data_y_list,tmp_y],0)
                    data_f_list = np.concatenate([data_f_list,tmp_factor],0)
            else:
                if len(zs_x_list)==0:
                    zs_x_list = tmp_images
                    zs_y_list = tmp_y
                    zs_f_list = tmp_factor
                else:
                    zs_x_list = np.concatenate([zs_x_list,tmp_images],0)
                    zs_y_list = np.concatenate([zs_y_list,tmp_y],0)
                    zs_f_list = np.concatenate([zs_f_list,tmp_factor],0)                
                    
    save_path = os.path.join(path,'color_mnist.npz')
    np.savez(save_path,
            t_x = data_x_list,
            t_y = data_y_list,
            t_f = data_f_list,
            zs_x = zs_x_list,
            zs_y = zs_y_list,
            zs_f = zs_f_list
            )



# ================= Generate linear map data ==================================
def _generate_MLP_Map(F_MAPPING, G_MAPPING, batch_size=64, validation_split=.2, random_seed=42,
                        x_dim=100, samples=100, noise=0):
    """
        Set the zero-shot table ZS_TABLE in configs.py
        When generating the data, first specify y = g(G1G2G3) and x = f(G1G2G3)
        Here function f() is F_MAPPING, a MLP or a matrix, g() is G_MAPPING
        The output contains 3 dataloader, each loader has three attributes:
            x: data, x = f(G1G2G3)
            y: target, y = g(G1G2G3)
            f: true factor, G1G2G3
        The permutation operation is on all factors G1G2G3, it only used to see
        the learning speed advantage.
    """    
    validation_split = validation_split
    batch_size = batch_size
    
    
    data_x_list = []
    data_y_list = []       # target is y = g(G1G2G3)         
    data_f_list = []       # true factor is G1G2G3
    zs_x_list = []
    zs_y_list = []
    zs_f_list = []
        
    for g1 in G1:
        for g2 in G2:
            if (g1, g2) not in ZS_TABLE:
                tmp_factor = torch.tensor([g1,g2])
                tmp_onehot = torch.zeros(NG1+NG2)  
                tmp_onehot[g1] = 1
                tmp_onehot[NG1+g2] = 1
                tmp_x = F_MAPPING(tmp_onehot).detach()
                tmp_y = G_MAPPING(tmp_onehot).detach()
                for i in range(samples):
                    noise_x = tmp_x + torch.randn(1,x_dim)*noise
                    data_x_list.append(noise_x)
                    data_y_list.append(tmp_y)
                    data_f_list.append(tmp_factor)
            else:
                tmp_factor = torch.tensor([g1,g2])
                tmp_onehot = torch.zeros(NG1+NG2)  
                tmp_onehot[g1] = 1
                tmp_onehot[NG1+g2] = 1
                tmp_x = F_MAPPING(tmp_onehot).detach()
                tmp_y = G_MAPPING(tmp_onehot).detach()
                for i in range(samples):
                    noise_x =tmp_x + torch.randn(1,x_dim)*noise
                    zs_x_list.append(noise_x)
                    zs_y_list.append(tmp_y)
                    zs_f_list.append(tmp_factor)                    
    
    x = torch.stack(data_x_list).reshape(-1,x_dim)          # [N, x_dim]
    y = torch.stack(data_y_list).int().reshape(-1,2)                # [N, 3], i.e., the true generating factor
    f = torch.stack(data_f_list).int().reshape(-1,2)         # [N, 27]
    dataset=Data.TensorDataset(x, y, f)
    
    if len(zs_x_list) > 0:
        zs_x = torch.stack(zs_x_list).reshape(-1,x_dim)
        zs_y = torch.stack(zs_y_list).int().reshape(-1,2)
        zs_f = torch.stack(zs_f_list).int().reshape(-1,2)
    else:
        zs_x = x
        zs_y = y
        zs_f = f

    zs_dataset = Data.TensorDataset(zs_x, zs_y, zs_f)
    
    np.random.seed(random_seed)
    dataset_size = len(data_x_list)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)     
                
    train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last = False)
    validation_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last = False)
    zs_loader = Data.DataLoader(zs_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    return train_loader, validation_loader, zs_loader

def _G_MAPPING_G12(tmp_onehot):
    g1 = tmp_onehot[:NG1].argmax()
    g2 = tmp_onehot[NG1:].argmax()
    return torch.tensor((g1,g2))

def _G_MAPPING_G12_MNIST(tmp_onehot):
    g1 = tmp_onehot[:,:NG1].argmax(1)
    g2 = tmp_onehot[:,NG1:].argmax(1)
    return torch.tensor((g1,g2)).transpose(0,1)


def Data_Gen_Twohots_Map(batch_size=64, validation_split=.2, random_seed=42, x_dim=20, samples=100, noise=0):
    
    def F_MAPPING_TWO_HOTS(tmp_onehot):
        return tmp_onehot
    
    t_loader, v_loader, zs_loader = _generate_MLP_Map(F_MAPPING_TWO_HOTS, _G_MAPPING_G12,
                                          batch_size=batch_size, 
                                          validation_split=validation_split, 
                                          random_seed=random_seed, 
                                          x_dim=x_dim, 
                                          samples=samples, 
                                          noise=noise)
    
    return t_loader, v_loader, zs_loader

def Data_Gen_Linear_Map(batch_size=64, validation_split=.2, random_seed=42,
                        x_dim=100, samples=100, noise=0):
    
    In_dim = G1.size(0)+G2.size(0)
    F_MAPPING = nn.Sequential(nn.Linear(In_dim,x_dim))
    for block in F_MAPPING:
        if isinstance(block, (nn.Linear)):
            init.normal_(block.weight, 0, 1)
            if block.bias is not None:
                block.bias.data.fill_(0.1)    

    t_loader, v_loader, zs_loader = _generate_MLP_Map(F_MAPPING, _G_MAPPING_G12,
                                          batch_size=batch_size, 
                                          validation_split=validation_split, 
                                          random_seed=random_seed, 
                                          x_dim=x_dim, 
                                          samples=samples, 
                                          noise=noise)
    
    return t_loader, v_loader, zs_loader
    

def Data_Gen_NonLinear_Map(batch_size=64, validation_split=.2, random_seed=42,
                        x_dim=100, samples=100, noise=0):
    In_dim = G1.size(0)+G2.size(0)
    F_MAPPING = nn.Sequential(
            nn.Linear(In_dim,64),
            nn.ReLU(True),
            nn.Linear(64,x_dim)
            )
    for block in F_MAPPING:
        if isinstance(block, (nn.Linear)):
            init.normal_(block.weight, 0, 0.25)
            if block.bias is not None:
                block.bias.data.fill_(0.1)

    t_loader, v_loader, zs_loader = _generate_MLP_Map(F_MAPPING, _G_MAPPING_G12,
                                          batch_size=batch_size, 
                                          validation_split=validation_split, 
                                          random_seed=random_seed, 
                                          x_dim=x_dim, 
                                          samples=samples, 
                                          noise=noise)
    
    return t_loader, v_loader, zs_loader


def Data_Gen_Color_MNIST(batch_size=64, validation_split=.2, random_seed=42):
    from pathlib import Path
    subpath = os.getcwd()
    if not Path(subpath+'/data/color_mnist.npz').exists():
        _generate_Color_MNIST(_G_MAPPING_G12_MNIST)
    
    np_data = np.load('data/color_mnist.npz')
    
    x = torch.tensor(np_data['t_x'])             # [N, 28,28,3]
    y = torch.tensor(np_data['t_y']).int()                  # [N, 3], i.e., the true generating factor
    f = torch.tensor(np_data['t_f']).int()
    if len(np_data['zs_x'])!=0:
        zs_x = torch.tensor(np_data['zs_x'])
        zs_y = torch.tensor(np_data['zs_y']).int()
        zs_f = torch.tensor(np_data['zs_f']).int()
    else:
        zs_x, zs_y, zs_f = x, y, f
    
    dataset=Data.TensorDataset(x, y, f)
    zs_dataset = Data.TensorDataset(zs_x, zs_y, zs_f)
    
    np.random.seed(random_seed)
    dataset_size = x.size(0)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)     
                
    train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last = True)
    validation_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last = True)
    zs_loader = Data.DataLoader(zs_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return train_loader, validation_loader, zs_loader

def y_to_ID(y):
    return y[:,0]+y[:,1]*NG1

def ID_to_y(ID):
    y2 = int(ID/NG1)
    y1 = int(ID - y2*NG1)
    return torch.tensor([y1, y2])








