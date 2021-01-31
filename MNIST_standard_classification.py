#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:52:33 2021

@author: joshua
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from matplotlib import pyplot as plt

from funcs.data_gen import *
from funcs.models import MLP_LSA, LeNet, MLP_classification
import funcs.utils
from funcs.configs import *
import warnings 
import random
warnings.filterwarnings('ignore')
def seed_torch(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()


CE_LOSS = nn.CrossEntropyLoss()
BCE_LOSS = nn.BCELoss()
MSE_LOSS = nn.MSELoss()
SFTMX0 = nn.Softmax(0)
SFTMX1 = nn.Softmax(1)
OUT_DIM_CLAS = NG1+NG2


def new_pop_CLAS(popsize):
    population = []
    for i in range(popsize):
        pop = LeNet(out_dim=OUT_DIM_CLAS, hid_size=HID_SIZE).cuda()
        population.append(pop)
    return population  

def pop_interact_clas(agent_, train_loader, valid_loader, all_x, all_y, rounds=1000, topsim_flag=False, print_rnd=True,lr=1e-3, lr_bob=1):
    agent = copy.deepcopy(agent_)
    optim_agent = optim.Adam(agent.parameters(),lr=lr)
    results={'t_loss':[],'t_acc':[],'v_loss':[],'v_acc':[],'data':[],'data_acc':[],'topsim_msg':[],'topsim_mid':[]}
    rnd_idx = 0
    inner_idx = 0
    # =============== Phase 1, supervised training and get pre-trained agent ======
    while (rnd_idx<rounds):
        rnd_idx += 1
        if print_rnd:
            print(str(rnd_idx)+"-",end="") 
        cor_cnt = 0
        all_cnt = 0
        for x, y, _ in train_loader:
            inner_idx += 1
            if topsim_flag and inner_idx %10 == 1:
                topsim_msg, topsim_mid = get_topsim_G12(agent, all_x)
                results['topsim_msg'].append(topsim_msg)
                results['topsim_mid'].append(topsim_mid)
            optim_agent.zero_grad()
            x = x.cuda().float().transpose(1,3)
            y = y.cuda().long()
            hid = agent(x)
            pred_loss = CE_LOSS(hid[:,:NG1], y[:,0]) + CE_LOSS(hid[:,NG1:], y[:,1])
            results['t_loss'].append(pred_loss.data.item())     
            pred_loss.backward()
            optim_agent.step()
            
            rwd_mask = get_rwd_from_hid_y(hid,y)
            cor_cnt += rwd_mask.sum()
            all_cnt += rwd_mask.shape[0]
        valid_results = get_valid_score(agent, valid_loader)
        results['v_loss'].append(valid_results['v_loss'])
        results['v_acc'].append(valid_results['v_acc'])
        results['t_acc'].append(float(cor_cnt)/float(all_cnt))
#    # ========== Phase 2, generate data after all interaction =======
#    gen_data_x = []
#    gen_data_mid = []
#    mid_size = mid.shape[1]
#    cor_cnt = 0
#    all_cnt = 0
#    
#    hid, mid = agent(all_x)
#    reward_mask = get_rwd_from_hid_y(hid,all_y)
#    data_mid = []
#    noise_flag = np.random.uniform(0,1)
#    for j in range(reward_mask.shape[0]):
#        all_cnt += 1
#        if noise_flag<0.01 or not reward_mask[j]:
#            rnd_mid = torch.tensor(np.random.randint(0,2,(mid_size,))).cuda().float()
#            data_mid.append(rnd_mid)    
#        else:
#            cor_cnt += 1
#            data_mid.append((mid[j,:]>0.5).float()) 
#            #data_mid.append(mid[j,:])               
#    gen_data_x = all_x
#    gen_data_mid = torch.stack(data_mid).reshape(-1,mid_size)
#    results['data'] = (gen_data_x, gen_data_mid)
#    results['data_acc'] = float(cor_cnt)/float(all_cnt)
    return agent, results



def pop_interact_Bob(agent_, train_loader, valid_loader, all_x, all_y, rounds=1000, topsim_flag=False, print_rnd=True,lr=1e-3):
    '''
        Freeze the parameters of Alice, use mid>0.5, and train Bob to match that
    '''
    batch_size = 10
    agent = copy.deepcopy(agent_)
    optim_agent_B = optim.Adam(agent.Bob.parameters(),lr=lr)
    results={'t_loss':[],'t_acc':[],'v_loss':[],'v_acc':[],'data':[],'data_acc':[],'topsim_msg':[],'topsim_mid':[]}
    rnd_idx = 0
    inner_idx = 0
    # =============== Phase 1, supervised training and get pre-trained agent ======
    while (rnd_idx<rounds):
        rnd_idx += 1
        if print_rnd:
            print(str(rnd_idx)+"-",end="") 
        cor_cnt = 0
        all_cnt = 0
        if topsim_flag and rnd_idx %4 == 1:
            topsim_msg, topsim_mid = get_topsim_G12(agent, all_x)
            results['topsim_msg'].append(topsim_msg)
            results['topsim_mid'].append(topsim_mid)
        optim_agent_B.zero_grad()
        
        data_idx = np.random.randint(0,all_x.shape[0],(batch_size,))
        x = all_x[data_idx]
        y = all_y[data_idx]
        x = x.cuda()
        y = y.cuda().long()
        mid = agent.Alice(x)
        mid_hard = (mid>0.5).float()
        hid = agent.Bob(mid_hard)
        pred_loss = CE_LOSS(hid[:,:NG1], y[:,0]) + CE_LOSS(hid[:,NG1:], y[:,1])
        results['t_loss'].append(pred_loss.data.item())     
        pred_loss.backward()
        optim_agent_B.step()
        
        rwd_mask = get_rwd_from_hid_y(hid,y)
        cor_cnt += rwd_mask.sum()
        all_cnt += rwd_mask.shape[0]
        valid_results = get_valid_score(agent, valid_loader)
        results['v_loss'].append(valid_results['v_loss'])
        results['v_acc'].append(valid_results['v_acc'])
        results['t_acc'].append(float(cor_cnt)/float(all_cnt))
    # ========== Phase 2, generate data after all interaction =======
    gen_data_x = []
    gen_data_mid = []
    mid_size = mid.shape[1]
    cor_cnt = 0
    all_cnt = 0
    
    hid, mid = agent(all_x)
    reward_mask = get_rwd_from_hid_y(hid,all_y)
    data_mid = []
    noise_flag = np.random.uniform(0,1)
    for j in range(reward_mask.shape[0]):
        all_cnt += 1
        if noise_flag<0.01 or not reward_mask[j]:
            rnd_mid = torch.tensor(np.random.randint(0,2,(mid_size,))).cuda().float()
            data_mid.append(rnd_mid)    
        else:
            cor_cnt += 1
            data_mid.append((mid[j,:]>0.5).float())                
    gen_data_x = all_x
    gen_data_mid = torch.stack(data_mid).reshape(-1,mid_size)
    results['data'] = (gen_data_x, gen_data_mid)
    results['data_acc'] = float(cor_cnt)/float(all_cnt)
    return agent, results




def pop_interact_clas_reinforce(agent_, train_loader, valid_loader, all_x, rounds=1000, topsim_flag=False, print_rnd=True,lr=1e-3):
    agent = copy.deepcopy(agent_)
    optim_agent_A = optim.Adam(agent.Alice.parameters(),lr=lr)
    optim_agent_B = optim.Adam(agent.Bob.parameters(),lr=lr)
    results={'t_loss':[],'t_acc':[],'v_loss':[],'v_acc':[],'data':[],'data_acc':[],'topsim_msg':[],'topsim_mid':[]}
    rnd_idx = 0
    inner_idx = 0
    # =============== Phase 1, supervised training and get pre-trained agent ======
    while (rnd_idx<rounds):
        rnd_idx += 1
        if print_rnd:
            print(str(rnd_idx)+"-",end="") 
        cor_cnt = 0
        all_cnt = 0
        for x, y, _ in train_loader:
            inner_idx += 1
            if topsim_flag and inner_idx %10 == 1:
                topsim_msg, topsim_mid = get_topsim_G12(agent, all_x)
                results['topsim_msg'].append(topsim_msg)
                results['topsim_mid'].append(topsim_mid)
            optim_agent_A.zero_grad()
            optim_agent_B.zero_grad()
            x = x.cuda()
            y = y.cuda().long()
            if inner_idx < 3000:
                hid, mid = agent(x, False)
            else:
                if rnd_idx%10==1:
                    agent.temp = agent.temp*0.99
                hid, mid = agent(x, True)
            B_loss = CE_LOSS(hid[:,:NG1], y[:,0]) + CE_LOSS(hid[:,NG1:], y[:,1])
            B_loss_item = B_loss.data
            results['t_loss'].append(B_loss_item.item())
            rwd_mask = get_rwd_from_hid_y(hid,y)
#            Alice_log_prob = mid.log().sum(1)
#            #A_loss = (-torch.clamp(B_loss_item,-2,2)*Alice_log_prob).sum()
#            A_loss = (-Alice_log_prob).sum()            
#            A_loss.backward()
            B_loss.backward()
            optim_agent_A.step()
            optim_agent_B.step()
            
            cor_cnt += rwd_mask.sum()
            all_cnt += rwd_mask.shape[0]
        valid_results = get_valid_score(agent, valid_loader)
        results['v_loss'].append(valid_results['v_loss'])
        results['v_acc'].append(valid_results['v_acc'])
        results['t_acc'].append(cor_cnt/all_cnt)
    # ========== Phase 2, generate data after all interaction =======
    gen_data_x = []
    gen_data_mid = []
    mid_size = mid.shape[1]
    cor_cnt = 0
    all_cnt = 0
    
    hid, mid = agent(all_x)
    reward_mask = get_rwd_from_hid_y(hid,all_y)
    data_mid = []
    noise_flag = np.random.uniform(0,1)
    for j in range(reward_mask.shape[0]):
        all_cnt += 1
        if noise_flag<0.01 or not reward_mask[j]:
            rnd_mid = torch.tensor(np.random.randint(0,2,(mid_size,))).cuda().float()
            data_mid.append(rnd_mid)    
        else:
            cor_cnt += 1
            data_mid.append((mid[j,:]>0.5).float())                
    gen_data_x = all_x
    gen_data_mid = torch.stack(data_mid).reshape(-1,mid_size)
    results['data'] = (gen_data_x, gen_data_mid)
    results['data_acc'] = cor_cnt/all_cnt
    return agent, results



def pop_train_from_data(agent_, data,all_x, batch_size=1, rounds=1000, topsim_flag=False, print_rnd=True):
    agent = copy.deepcopy(agent_)
    optim_agent_A = optim.Adam(agent.Alice.parameters(),lr=1e-3)
    x = data[0]
    tgt_mid = data[1].detach()
    data_length = x.shape[0]
    results = {'t_loss':[],'topsim_msg':[],'topsim_mid':[]}
    
    agent.train()   
    rnd_idx = 0
    while (rnd_idx<rounds):
        rnd_idx += 1
        if print_rnd:
            print(str(rnd_idx)+"-",end="") 
        if topsim_flag and rnd_idx %2 == 1:
            topsim_msg, topsim_mid = get_topsim_G12(agent, all_x)
            results['topsim_msg'].append(topsim_msg)
            results['topsim_mid'].append(topsim_mid)
        optim_agent_A.zero_grad()
        data_idx = np.random.randint(0,data_length,(batch_size,))
        hid, mid = agent(x[data_idx])
        learn_loss = BCE_LOSS(mid, tgt_mid[data_idx])
#        learn_loss = MSE_LOSS(mid, tgt_mid[data_idx])
        learn_loss.backward()
        optim_agent_A.step()  
        results['t_loss'].append(learn_loss.data.item())
    return agent, results        


def pop_train_from_agent(learner_, teacher, data_loader, all_x, rounds=1000, topsim_flag=False, print_rnd=True, lr=1e-4):
    learner = copy.deepcopy(learner_)
    optim_agent_A = optim.Adam(learner.Alice.parameters(),lr=lr)
    results = {'t_loss':[],'topsim_msg':[],'topsim_mid':[],'gen_ability':[],'gen_acc_test':[]}
    learner.train()
    teacher.train()
    rnd_idx = 0
    inner_idx = 0
    while (rnd_idx<rounds):
        rnd_idx += 1
        if print_rnd:
            print(str(rnd_idx)+"-",end="") 
        for x, _, _ in data_loader:
            inner_idx += 1
            if topsim_flag and inner_idx %50 == 1:
                topsim_msg, topsim_mid = get_topsim_G12(learner, all_x)
#                tmp_results = get_gen_ability_of_agent(learner, rounds=20)
#                results['gen_acc_test'].append(np.mean(tmp_results['t_acc'][-10:]))
#                results['gen_ability'].append(tmp_results['v_acc'])
                results['topsim_msg'].append(topsim_msg)
                results['topsim_mid'].append(topsim_mid)
            optim_agent_A.zero_grad()
            x = x.cuda()   
            # === Here is full-batch training, change to sub-batch later
            teach_pred, teach_mid = teacher(x)
            teach_mid = teach_mid.detach()
            learn_pred, learn_mid = learner(x)
            
            teach_tgt = (teach_mid>0.5).float()            
            learn_loss = BCE_LOSS(learn_mid, teach_tgt)
#            learn_loss = MSE_LOSS(learn_mid, teach_mid)
            
            results['t_loss'].append(learn_loss.data.item())
            learn_loss.backward()
            optim_agent_A.step()  
    return learner, results


def get_valid_score(agent_, valid_loader):
    agent = copy.deepcopy(agent_)
    agent.eval()
    results={'v_loss':[],'v_acc':[]}
    loss_table = []
    cor_cnts = 0
    all_cnts = 0
    with torch.no_grad():
        for x, y, _ in valid_loader:
            x = x.cuda().float().transpose(1,3)
            y = y.cuda().long()
            hid_pred = agent(x)
            y1_pred = hid_pred[:,:NG1].argmax(1)
            y2_pred = hid_pred[:,NG1:].argmax(1)
            y_pred = torch.cat((y1_pred.unsqueeze(1),y2_pred.unsqueeze(1)),axis=1)            
            valid_loss = CE_LOSS(hid_pred[:,:NG1], y[:,0]) + CE_LOSS(hid_pred[:,NG1:], y[:,1])
            loss_table.append(valid_loss.data.item())
            
            cor_cnts += ((y_pred==y).sum(1)==2).cpu().sum().float()
            all_cnts += float(x.shape[0])
    results['v_loss'] = np.mean(loss_table)
    results['v_acc'] = cor_cnts/all_cnts
    agent.train()
    return results

def get_rwd_from_hid_y(hid,y):
    hid = hid.cpu().float()
    y = y.cpu().long()
    y1_pred = hid[:,:NG1].argmax(1)
    y2_pred = hid[:,NG1:].argmax(1)
    reward_mask = ((y1_pred==y[:,0]).float() + (y2_pred==y[:,1]).float())==2  
    return reward_mask

def get_allxyID(data_loader):
    all_x = []
    all_y = []
    all_ID = []
    seen_ID = []
    for x, y, _ in data_loader:
        for i in range(x.shape[0]):
            ID = y[i,0]*NG1+y[i,1] #+y[i,2]*NG1*NG2
            if ID not in seen_ID:
                all_x.append(x[i])
                all_y.append(y[i])
                all_ID.append(ID)
                seen_ID.append(ID)
            else:
                pass
    all_x = torch.stack(all_x).cuda()
    all_y = torch.stack(all_y).reshape(all_x.shape[0],-1).cuda()
    all_y = all_y.long()
    all_ID = torch.stack(all_ID).reshape(all_x.shape[0],-1).cuda()
    
    _, sort_idxs = all_ID.sort(0,False)
    sort_idxs = sort_idxs.squeeze()
    order_all_x = all_x[sort_idxs]
    order_all_y = all_y[sort_idxs]
    order_all_ID = all_ID[sort_idxs]
    return all_x, all_y, all_ID, order_all_x, order_all_y, order_all_ID

def _cal_dist(v1, v2, dis_type='hamming'):
    '''
        v1 and v2 should be two vectors with same length
        dis_type should be `hamming` or `cosine`
    '''
    v1 = v1.reshape(1,-1)
    v2 = v2.reshape(1,-1)
    if dis_type.lower()=='hamming':
        return torch.cdist(v1.float(),v2.float(),p=0)[0]
    elif dis_type.lower()=='cosine':
        return -nn.functional.cosine_similarity(v1,v2, dim=1)
    elif dis_type.lower()=='eu':
        pdist = nn.PairwiseDistance(p=2)
        return pdist(v1,v2)
    else:
        print('dis_type should be `hamming` or `cosine` or `eu`') 

def get_topsim_data(all_x,all_y):
    x = all_x.cpu()
    y = all_y.cpu()
    xdist_pair, ydist_pair = [],[]
    for i in range(all_x.shape[0]):
        for j in range(i):
            xdist_pair.append(_cal_dist(x[i,:],x[j,:],'cosine'))
            ydist_pair.append(_cal_dist(y[i,:],y[j,:],'hamming'))
    return pearsonr(xdist_pair, ydist_pair)

def get_topsim_G12(agent_, all_x):
    agent=copy.deepcopy(agent_)
    with torch.no_grad():
        data_len = all_x.shape[0]
        hid, mid = agent(all_x)
        msg1 = hid[:,:NG1].argmax(1).unsqueeze(1)
        msg2 = hid[:,NG1:].argmax(1).unsqueeze(1)
        msg = torch.cat((msg1,msg2),axis=1)
        x = all_x.cpu().float()
        msg = msg.cpu().float()
        mid = mid.cpu().float()
        mid = (mid>0.5).float()
        xdist_pair, msgdist_pair, middist_pair = [], [],[]
        for i in range(data_len):
            for j in range(i):
                xdist_pair.append(_cal_dist(x[i,:],x[j,:],'cosine'))
                msgdist_pair.append(_cal_dist(msg[i,:],msg[j,:],'hamming'))
                middist_pair.append(_cal_dist(mid[i,:],mid[j,:],'hamming'))
        msgdist_pair[-1] = 0.01 
        middist_pair[-1] = 0.01
        x_to_msg = pearsonr(xdist_pair,msgdist_pair)[0]
        x_to_mid = pearsonr(xdist_pair,middist_pair)[0]

        return x_to_msg, x_to_mid
    
def get_gen_ability_of_agent(agent_, rounds=10):
    '''
        After pre-train, we fix Alice and update the parameters of Bob on training
        set. After convergence, we see the validation performance of Alice+Bob
    '''
    results = {'t_loss':[],'t_acc':[],'v_loss':[],'v_acc':[]}
    agent = copy.deepcopy(agent_)
    # ========= Train some rounds on training set ====================
    for i in range(rounds):
        optim_agent_B = optim.Adam(agent.Bob.parameters(),lr=1e-3)
        for x, y, _ in train_loader:
            x = x.cuda()
            y = y.cuda().long()
            hid, mid = agent(x)
            mid_hard = (mid>0.5).float()
            #mid_hard = mid.detach()
            hid_hard = agent.Bob(mid_hard)
            pred_loss = CE_LOSS(hid_hard[:,:NG1], y[:,0]) + CE_LOSS(hid_hard[:,NG1:], y[:,1])
            results['t_loss'].append(pred_loss.data.item())
            
            optim_agent_B.zero_grad()
            pred_loss.backward()
            optim_agent_B.step()
            rwd_hard = get_rwd_from_hid_y(hid_hard,y)
            results['t_acc'].append(rwd_hard.sum())
            # ========= Calculate validation score on validation set =========
#        valid_results = get_valid_score(agent, zs_loader)
#        results['v_acc'].append(valid_results['v_acc'])
#        results['v_loss'].append(valid_results['v_loss'])
    valid_results = get_valid_score(agent, zs_loader)
    results['v_acc'] = valid_results['v_acc']
    results['v_loss'] = valid_results['v_loss']
    return results


PRE_ROUNDS = 100
INT_ROUNDS = 200

train_loader, valid_loader, zs_loader = Data_Gen_Color_MNIST(batch_size=10, validation_split=0.2,random_seed=42)
all_x, all_y, all_ID, order_all_x, order_all_y, order_all_ID = get_allxyID(train_loader)
#all_vx, all_vy, all_vID = get_allxyID(zs_loader)

pop_infa = new_pop_CLAS(1)[0]
pop_adul, inter_results = pop_interact_clas(pop_infa, train_loader, valid_loader, all_x,all_y, rounds=INT_ROUNDS, topsim_flag=False,lr=5e-4) 
#data_xmid = inter_results['data']

#all_img = order_all_x.cpu()
#big_images = np.zeros((280,280,3))
#for i in range(10):
#    for j in range(10):
#        if (i*10+j)<65:
#            big_images[i*28:(i+1)*28,j*28:(j+1)*28,:] = all_img[i*10+j].float()/255
#        else:
#            big_images[i*28:(i+1)*28,j*28:(j+1)*28,:] = all_img[i*10+j].float()/255
#plt.imshow(big_images)   
#








