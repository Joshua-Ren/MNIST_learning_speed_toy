#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:24:58 2020

@author: joshua
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from funcs.data_gen import Data_Gen_Linear_Map, Data_Gen_NonLinear_Map, Data_Gen_Color_MNIST
from funcs.models import MLP, LeNet
import funcs.utils
from funcs.configs import *
import warnings 
warnings.filterwarnings('ignore')

out_dim = NG1+NG2+NG3

"""

train_loader, valid_loader = Data_Gen_Color_MNIST(batch_size=64, validation_split=.9, random_seed=42)

net = LeNet(out_dim=out_dim).cuda()
optimizer = optim.Adam(net.parameters(),lr=1e-4)
LOSS_FUN = nn.CrossEntropyLoss()


loss_table = []
valid_table = []
for i_ep in range(10):
    print(i_ep)
    net.train()
    for b_id, (x,y_origin,_) in enumerate(train_loader):
        x = x.float().cuda()
        y1 = torch.tensor(y_origin[:,0],dtype=torch.long).cuda()
        y2 = torch.tensor(y_origin[:,1],dtype=torch.long).cuda()
        y3 = torch.tensor(y_origin[:,2],dtype=torch.long).cuda()
        y_hidden = net(x)
        loss = LOSS_FUN(y_hidden[:,:NG1],y1) + LOSS_FUN(y_hidden[:,NG1:NG1+NG2],y2) + LOSS_FUN(y_hidden[:,NG1+NG2:],y3)
        #loss = LOSS_FUN(y_hidden[:,:10],y1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        loss_table.append(loss.data.item())  
        #print(loss.data.item())
    
    net.eval()
    tested_samples = 0
    correct_samples = 0
    for b_id, (x,y_origin,_) in enumerate(valid_loader):       
        x = x.float().cuda()
        y1_target = torch.tensor(y_origin[:,0],dtype=torch.long).cuda()
        y2_target = torch.tensor(y_origin[:,1],dtype=torch.long).cuda()
        y3_target = torch.tensor(y_origin[:,2],dtype=torch.long).cuda()
        y_hidden = net(x)
        y1_predict = y_hidden[:,:10].argmax(dim=1)
        y2_predict = y_hidden[:,10:19].argmax(dim=1)
        y3_predict = y_hidden[:,19:].argmax(dim=1)
        
        num_samples = y_origin.size(0)
        tested_samples += num_samples
        correct_samples += (y1_target==y1_predict).sum().cpu().float()
        correct_samples += (y2_target==y2_predict).sum().cpu().float()
        correct_samples += (y3_target==y3_predict).sum().cpu().float()
    valid_table.append(correct_samples/tested_samples/3)

plt.figure(1)
plt.plot(valid_table)    
plt.figure(2)
plt.plot(loss_table)
plt.ylim(0,2)

#plt.imshow(x[0].transpose(0,2).cpu()/255)




"""
train_loader, valid_loader = Data_Gen_NonLinear_Map(batch_size=64, 
                                                    validation_split=0.5, 
                                                    random_seed=42, 
                                                    x_dim=100, 
                                                    samples=100, 
                                                    noise=0.05,
                                                    ratio_perm=0)

net = MLP(in_dim=100, out_dim=out_dim).cuda()
optimizer = optim.Adam(net.parameters(),lr=5e-4)
LOSS_FUN = nn.CrossEntropyLoss()

loss_table = []
valid_table = []
for i_ep in range(500):
    net.train()
    for b_id, (x,y_origin,_) in enumerate(train_loader):
        x = x.cuda()
        y1 = torch.tensor(y_origin[:,0],dtype=torch.long).cuda()
        y2 = torch.tensor(y_origin[:,1],dtype=torch.long).cuda()
        y3 = torch.tensor(y_origin[:,2],dtype=torch.long).cuda()
        y_hidden = net(x)
        loss = LOSS_FUN(y_hidden[:,:NG1],y1) + LOSS_FUN(y_hidden[:,NG1:NG1+NG2],y2) + LOSS_FUN(y_hidden[:,NG1+NG2:],y3)
        #loss = LOSS_FUN(y_hidden[:,:10],y1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        loss_table.append(loss.data.item())  
        #print(loss.data.item())
    
    net.eval()
    tested_samples = 0
    correct_samples = 0
    for b_id, (x,y_origin,_) in enumerate(valid_loader):       
        x = x.cuda()
        y1_target = torch.tensor(y_origin[:,0],dtype=torch.long).cuda()
        y2_target = torch.tensor(y_origin[:,1],dtype=torch.long).cuda()
        y3_target = torch.tensor(y_origin[:,2],dtype=torch.long).cuda()
        y_hidden = net(x)
        y1_predict = y_hidden[:,:NG1].argmax(dim=1)
        y2_predict = y_hidden[:,NG1:NG1+NG2].argmax(dim=1)
        y3_predict = y_hidden[:,NG1+NG2:].argmax(dim=1)
        
        num_samples = y_origin.size(0)
        tested_samples += num_samples
        correct_samples += (y1_target==y1_predict).sum().cpu().float()
        correct_samples += (y2_target==y2_predict).sum().cpu().float()
        correct_samples += (y3_target==y3_predict).sum().cpu().float()
    valid_table.append(correct_samples/tested_samples/3)
        
plt.figure(1)
plt.plot(valid_table)    
plt.figure(2)
plt.plot(loss_table)
plt.ylim(0,2)

