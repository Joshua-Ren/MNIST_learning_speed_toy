#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:46:18 2020
Here we may have several fundamental models in deep learning, e.g., MLP, CNN,...
We can use the same model for both regression and classification tasks, only 
change the value of out_dim.

For classification task, the out vector is logprob, we should use CrossEntropy 
loss during training. For the regression task, the out vector is y_hat, we can
use L2 loss.

@author: joshua
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from .configs import *
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
         
class MLP(nn.Module):
    '''
        We can use the same MLP modual when doing regression and classification
        problems. In classification problem, the output of the modual is 
        logprob (logit). We should use CrossEntropy loss. In regression problem,
        the output is just the prediction.
        
        Usually, the out_dim in classification problem is large while for regression
        is small.
        
        For G12 or G23, only need to change out_dim and use the loss function to
        split two attributes.
    '''
    def __init__(self, in_dim, out_dim, hid_size=128):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_size = hid_size
        
        self.fc1 = nn.Linear(self.in_dim, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.fc3 = nn.Linear(self.hid_size, self.hid_size)
        self.fc4 = nn.Linear(self.hid_size, self.out_dim)
        self.act = nn.LeakyReLU(0.2, True)
        
        self.tf_layer = nn.Sequential(
                nn.Linear(8,16),
                nn.LeakyReLU(0.2, True),
#                nn.Linear(16,16),
#                nn.LeakyReLU(0.2, True),
                nn.Linear(16,8),
                nn.LeakyReLU(0.2, True),
                nn.Linear(8,2)                
                )
        self.clas_layer = nn.Sequential(
                nn.Linear(4,8),
                nn.LeakyReLU(0.2, True),
                nn.Linear(8,4)
                )
    
    def forward(self,x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        out = self.fc4(h3)
        return out


class MLP_classification(nn.Module):
    '''
        For the standard classification task, here we want to design several 
        different MLP stuctures, and find a good way to separate Alice/Bob
    '''
    def __init__(self, in_dim, out_dim, hid_size=HID_SIZE, mid_size=MID_SIZE, temp=2):
        super(MLP_classification, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_size = hid_size
        self.mid_size = mid_size
        self.temp = temp
        
        self.Alice = nn.Sequential(
                nn.Linear(self.in_dim, self.hid_size),
                nn.ReLU(),
                nn.Linear(self.hid_size, self.hid_size),
                nn.ReLU(),
                nn.Linear(self.hid_size, self.mid_size),
                nn.Sigmoid()                # Core design part
                )
        self.Bob = nn.Sequential(
                nn.Linear(self.mid_size, self.hid_size),
                nn.ReLU(),
                nn.Linear(self.hid_size, self.hid_size),
                nn.ReLU(),
                nn.Linear(self.hid_size, self.out_dim)
                )
    def forward(self,x,rein_flag=False):
        h1 = self.Alice(x)
        if rein_flag:
            h_distribution = RelaxedBernoulli(self.temp, h1)
            h1_nograd = h_distribution.sample()        
            h1_nograd = h1_nograd.float()
            out = self.Bob(h1_nograd)
        else:
            out = self.Bob(h1)
        return out, h1


class MLP_LSA(nn.Module):
    '''
        We can use the same MLP modual when doing regression and classification
        problems. In classification problem, the output of the modual is 
        logprob (logit). We should use CrossEntropy loss. In regression problem,
        the output is just the prediction.
        
        Usually, the out_dim in classification problem is large while for regression
        is small.
        
        For G12 or G23, only need to change out_dim and use the loss function to
        split two attributes.
    '''
    def __init__(self, in_dim, out_dim, hid_size=128):
        super(MLP_LSA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_size = hid_size
        
        self.fc1 = nn.Linear(self.in_dim, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.fc3 = nn.Linear(self.hid_size, self.hid_size)
        self.fc4 = nn.Linear(self.hid_size, self.hid_size)
        self.fc5 = nn.Linear(self.hid_size, self.out_dim)
        self.act = nn.LeakyReLU(0.2, True)
        self.act2 = nn.Tanh()
        
        self.get_log_probs = 0
        self.get_entropy = 0
        
#        self.clas_layer = nn.Sequential(
#                nn.Linear(30,30)
##                nn.LeakyReLU(0.2,True),
##                nn.Linear(30,30)
#                )
    
    def forward(self,x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        h4 = self.act(self.fc4(h3))
        out = self.fc5(h4)
        probs1 = F.softmax(out[:,:NG1],dim=1)
        probs2 = F.softmax(out[:,NG1:],dim=1)  
        distr1 = OneHotCategorical(probs=probs1)
        distr2 = OneHotCategorical(probs=probs2)
        msg_oht1 = distr1.sample()
        msg_oht2 = distr2.sample()
        
        self.get_log_probs = torch.log((probs1*msg_oht1).sum(1))+torch.log((probs2*msg_oht2).sum(1))
        self.get_entropy = distr2.entropy()        
        msg1 = msg_oht1.argmax(1)
        msg2 = msg_oht2.argmax(1)
        msgs_value = torch.cat((msg1.unsqueeze(1),msg2.unsqueeze(1)),dim=1)       
        return out, msgs_value


class Alice_agent_LSTM(nn.Module):
    def __init__(self, in_dim, hid_size=64):
        super().__init__()
        self.in_dim = in_dim    # dim of X
        self.hid_size = hid_size
        self.out_size = NG1
        self.x_to_embd = nn.Sequential(
                nn.Linear(self.in_dim, self.hid_size),
                nn.ReLU(),
#                nn.Linear(self.hid_size, self.hid_size),    #add
#                nn.ReLU(),       #add
#                nn.Linear(self.hid_size, self.hid_size),    #add
#                nn.ReLU()       #add
                )
        self.lstm = nn.LSTMCell(NG1, self.hid_size)
        self.out_layer = nn.Linear(self.hid_size, self.out_size)
        self.get_log_probs = 0
        self.get_entropy = 0
        
    def forward(self, tgt_x):
        batch_size = tgt_x.shape[0]
        tgt_hid = self.x_to_embd(tgt_x)
        lstm_input = torch.zeros((batch_size,NG1)).cuda()
        lstm_hid = tgt_hid.squeeze(1)
        lstm_cell = tgt_hid.squeeze(1)
        msgs = []
        msgs_value =[]
        logits = []
        log_probs = 0.
        
        for _ in range(2):
            lstm_hid, lstm_cell = self.lstm(lstm_input, (lstm_hid,lstm_cell))
            logit = self.out_layer(lstm_hid)
            logits.append(logit)
            probs = nn.functional.softmax(logit, dim=1)
            if self.training:
                cat_distr = OneHotCategorical(probs=probs)
                msg_oht, entropy = cat_distr.sample(), cat_distr.entropy()
                self.get_entropy = entropy
            else:
                msg_oht = nn.functional.one_hot(torch.argmax(probs, dim=1),num_classes=self.out_size).float()
            log_probs += torch.log((probs*msg_oht).sum(1))
            msgs.append(msg_oht)
            msgs_value.append(msg_oht.argmax(1))
            lstm_input = msg_oht
            
        msgs = torch.stack(msgs)
        msgs_value = torch.stack(msgs_value).transpose(0,1)
        logits = torch.stack(logits)
        logits = logits.transpose(0,1).reshape(batch_size,-1)
        
        self.get_log_probs = log_probs
        return logits, msgs_value
            

  

class Bob_agent_LSTM(nn.Module):
    def __init__(self,in_dim, hid_size=64,candi_size=5):
        super(Bob_agent_LSTM,self).__init__()
        self.in_dim = in_dim
        self.hid_size = hid_size
        self.candi_size = candi_size
        self.lstm = nn.LSTMCell(NG1,self.hid_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()
        self.candi_to_h = nn.Sequential(
                nn.Linear(self.in_dim, self.hid_size),
#                nn.ReLU(),              #add
#                nn.Linear(self.hid_size, self.hid_size), #add
#                nn.ReLU(),              #add
#                nn.Linear(self.hid_size, self.hid_size) #add
            )
        self.hid_to_msghid = nn.Sequential(nn.Linear(self.hid_size,self.hid_size))

    def msg_to_twohots(self, msg):
        '''
            The msg is two int (range from 0 to NG1-1) values
            type of msg is [N_B, 2], output should be [N_B, NG1+NG2]
        '''
        twohots = []
        for i in range(msg.shape[0]):
            tmp_hots = torch.zeros((NG1+NG2,1)).cuda()
            idx1 = msg[i,0]
            idx2 = NG1+ msg[i,1]
            tmp_hots[idx1] = 1
            tmp_hots[idx2] = 1
            twohots.append(tmp_hots)
        return torch.stack(twohots).squeeze()
    
    def candis_to_hs(self, candis):
        candi_hs = []
        for i in range(self.candi_size):
            tmp_h = self.candi_to_h(candis[:,i,:])
            candi_hs.append(tmp_h)
        return torch.stack(candi_hs).squeeze().transpose(0,1)  #[N_B, N_candi, hid_size]
            
    def forward(self, msg, candis, gs_flag=False):
        batch_size = msg.shape[0]
        if gs_flag:
            twohots = msg
        else:
            twohots = self.msg_to_twohots(msg)  # --> [N_B, 20]
        two_onehots = twohots.reshape(batch_size,2,NG1).transpose(0,1)
        last_hidden = self.init_hidden.expand(batch_size,-1).contiguous()
        last_cell = self.init_cell.expand(batch_size,-1).contiguous()
        for t in range(2):
            hidden, cell = self.lstm(two_onehots[t],(last_hidden,last_cell))
            last_hidden = hidden
            last_cell =cell
        
        msghid = self.hid_to_msghid(last_hidden)
        msghid = msghid.unsqueeze(2) 
        candi_hs = self.candis_to_hs(candis)  
        candi_dot_hid = torch.bmm(candi_hs, msghid) # --> [N_B, hid_size]       
        #candi_dot_hid = F.cosine_similarity(candi_hs.transpose(1,2), msghid)        
        pred_vector = candi_dot_hid.squeeze()
        return pred_vector
    
    def init_hidden_and_cell(self):
        return torch.zeros(1, self.hid_size).cuda()

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class LeNet(nn.Module):
   def __init__(self, out_dim=NG1, hid_size=128):
       super(LeNet,self).__init__()
       self.Alice = nn.Sequential(
               nn.Conv2d(3, 6, 5, padding=2),
               nn.ReLU(),
               nn.MaxPool2d(1,(2,2)),
               nn.Conv2d(6, 16, 5),
               nn.ReLU(),
               nn.MaxPool2d(1,(2,2)),
               View((-1, 400)),               #16*5*5
#               nn.Tanh()                         # core design part               
               )
       self.Bob = nn.Sequential(
               nn.Linear(400, hid_size),
               nn.ReLU(),
               nn.Linear(hid_size, 84),
               nn.ReLU(),
               nn.Linear(84, out_dim)              
               )
       
   def forward(self, x):
       mid = self.Alice(x)
       hid = self.Bob(mid)
       return hid, mid
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features            




## ================= Backup of the original LeNet ===============
#class LeNet(nn.Module):
#   def __init__(self, out_dim=NG1, hid_size=128):
#       super(LeNet,self).__init__()
#       self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
#       self.conv2 = nn.Conv2d(6, 16, 5)
#       self.fc1 = nn.Linear(16*5*5, hid_size)
#       self.fc2 = nn.Linear(hid_size, 84)
#       self.fc3 = nn.Linear(84, out_dim)
#       self.tanh = nn.Tanh()
#       self.sigm = nn.Sigmoid()
#   def forward(self, x):
#       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#     
#       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#       mid = self.tanh(x.view(-1, self.num_flat_features(x)))
#       
#       x = F.relu(self.fc1(mid))
#       x = F.relu(self.fc2(x))
#       hid = self.fc3(x)
#       return hid, mid
#   def num_flat_features(self, x):
#       size = x.size()[1:]
#       num_features = 1
#       for s in size:
#           num_features *= s
#       return num_features    