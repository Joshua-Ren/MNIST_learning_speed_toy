#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:57:19 2020

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
from funcs.models import MLP_LSA, LeNet, Bob_agent_LSTM, Alice_agent_LSTM
import funcs.utils
from funcs.configs import *
import warnings 
import random
warnings.filterwarnings('ignore')


CE_LOSS = nn.CrossEntropyLoss()
MSE_LOSS = nn.MSELoss()
SFTMX0 = nn.Softmax(0)
SFTMX1 = nn.Softmax(1)
OUT_DIM_LSA = NG1+NG2


def new_pop_LSA(popsize):
    """
        We use probability distribution (256-length vector) to represent an agent
    """
    population = []
    OUT_DIM = NG1+NG2
    for i in range(popsize):
        pop = Alice_agent_LSTM(in_dim=X_DIM, hid_size=HID_SIZE).cuda()
        #pop = MLP_LSA(in_dim=X_DIM,out_dim=OUT_DIM).cuda()
        population.append(pop)
    return population  

def pop_train_from_data_LSA_SGD(agent, data, all_x, rounds, batch_size=1, see_valid=False,see_topsim=False, v_loader=None):
    results = {'loss':[],'v_acc':[],'v_loss':[],'topsim':[],'t_acc':[]}
    x = data[0]
    hh = data[1]
    optim_agent = optim.Adam(agent.parameters(),lr=1e-3)
    data_length = data[0].shape[0]
    rnd_idx = 0
    while (rnd_idx<rounds):
        if see_topsim and rnd_idx%(int(rounds*0.01))==0:
            topsim = _get_topsim_G12(agent,all_x)
            results['topsim'].append(topsim)
        if see_valid and rnd_idx%2==0:#int(rounds*0.02)==0:
            agent.eval()
            accuracy, v_loss = _get_valid_score_LSA_G12(agent, v_loader)
            results['v_acc'].append(accuracy)
            results['v_loss'].append(v_loss)
        agent.train()
        rnd_idx += 1
        data_idx = np.random.randint(0,data_length,(batch_size,))
        hidden, msg = agent(x[data_idx])
        loss = CE_LOSS(hidden.reshape(-1,2,NG1).transpose(1,2),hh[data_idx])
        #loss =  CE_LOSS(hidden[:,:NG1],hh[data_idx,0]) + CE_LOSS(hidden[:,NG1:],h[data_idx,1]) 
        optim_agent.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 50)
        optim_agent.step() 
        results['loss'].append(loss.data.item())         
        
        tmp_tacc = ((hh[data_idx] == msg).sum(1)==2).sum()
        results['t_acc'].append(tmp_tacc.item())       
    return agent, results


def pop_interact_refgame(speaker,all_x, rounds, batch_size=1, candi_size=5, see_valid=False, see_topsim=False, v_loader=None):
    
    listener = Bob_agent_LSTM(in_dim=X_DIM,hid_size=HID_SIZE,candi_size=candi_size)
    listener.cuda()
   
    results = {'spk_loss':[],'lis_loss':[],'accuracy':[],'v_loss':[], 
               'v_score':[], 'topsim':[],'ambi_ID':[],'rewards':[],'data':[]}
    
    optim_speaker = optim.Adam(speaker.parameters(),lr=1e-3)
    optim_listener = optim.Adam(listener.parameters(),lr=1e-3)
    rnd_idx = 0
    while (rnd_idx<rounds):
        rnd_idx += 1
        
        if see_topsim and rnd_idx%(int(rounds*0.01))==1:
            topsim = _get_topsim_G12(speaker,all_x)
            results['topsim'].append(topsim)
        if see_valid and rnd_idx%int(rounds*0.01)==1:
            speaker.eval()
            accuracy, v_loss = _get_valid_score_LSA_G12(speaker, v_loader)
            results['accuracy'].append(accuracy)
            results['v_loss'].append(v_loss)
            speaker.train()        

        speaker.train()
        listener.train()       
        optim_speaker.zero_grad()
        optim_listener.zero_grad()
        
        if rnd_idx%4==1:
             candis_tgtx, candis_allx, target_pos = _gen_candis(all_x, candi_size=candi_size)
        _, msg = speaker(candis_tgtx)
        spk_log_prob = speaker.get_log_probs
        spk_entropy = speaker.get_entropy   
        
        pred_vector = listener(msg, candis_allx, gs_flag=False)  
        
        #lis_log_prob = (SFTMX1(pred_vector)[torch.range(0,99).long(),target_pos]).log()
        lis_log_prob = nn.functional.log_softmax(pred_vector.max(1)[0])
        lis_entorpy = -(SFTMX1(pred_vector)*SFTMX1(pred_vector).log()).sum(dim=1)
        
        pred_pos = pred_vector.argmax(1).cpu()
        reward_mask = (target_pos-pred_pos)==0
        reward_mask = reward_mask.detach().int().cuda()
        
        results['rewards'].append(reward_mask.sum())

        lis_loss = -((lis_log_prob*(reward_mask)).mean()+0.05*lis_entorpy.mean())
        #lis_loss = CE_LOSS(pred_vector, target_pos.cuda().long())
        spk_loss = -((spk_log_prob*reward_mask).mean()+0.1*spk_entropy.mean())
        #spk_loss = lis_loss
        lis_loss.backward()  
        spk_loss.backward()        
        nn.utils.clip_grad_norm_(speaker.parameters(), 50)
        nn.utils.clip_grad_norm_(listener.parameters(), 50)  
        
        if rnd_idx < 200:   # Update listener only
            optim_speaker.zero_grad()
            optim_listener.step()    
        else:               # Update both        
            optim_speaker.step()
            optim_listener.step()        
        results['spk_loss'].append(spk_loss.data.item())
        results['lis_loss'].append(lis_loss.data.item())  
     
    # ================== get valid performance
    results['v_score'] = _get_valid_score_refgame(speaker, listener, v_loader, candi_size=candi_size)
    # ================= Directly get data from here =======    
    gen_data_x = []
    gen_data_msg = []    
    for i in range(100):
        candis_tgtx, candis_allx, target_pos = _gen_candis(all_x, candi_size=candi_size)
        with torch.no_grad():
            _, msg = speaker(candis_tgtx)
            pred_vector = listener(msg, candis_allx, gs_flag=False)  
            pred_pos = pred_vector.argmax(1).cpu()
            reward_mask = (target_pos-pred_pos)==0
            reward_mask = reward_mask.detach().int().cuda()
            data_msg = []
            for i in range(reward_mask.shape[0]):
                if reward_mask[i]==1:
                    data_msg.append(msg[i,:])
                else:
                    rnd_msg = torch.tensor(np.random.randint(0,NG1,(2,))).cuda()
                    data_msg.append(rnd_msg)
            gen_data_x.append(candis_tgtx)
            gen_data_msg.append(torch.stack(data_msg))
    gen_data_x = torch.stack(gen_data_x).reshape(-1,X_DIM)
    gen_data_msg = torch.stack(gen_data_msg).reshape(-1,2)
    results['data'] = (gen_data_x, gen_data_msg)        
    return speaker, results

def _gen_candis(all_x, candi_size=15):
    import copy
    data_length = all_x.shape[0]
    candi_idxs = np.zeros((data_length,candi_size))
    orign_smp_list = list(np.arange(0,data_length,1))
    target_pos = torch.zeros((data_length,),dtype=torch.long)
    target_idx = torch.zeros((data_length,),dtype=torch.long)
    
    for i in range(data_length):
        tmp_smp_list = copy.deepcopy(orign_smp_list)
        tmp_smp_list.remove(i)   
        candi_idxs[i,0] = i
        candi_idxs[i,1:] = random.sample(tmp_smp_list,candi_size-1)  
        np.random.shuffle(candi_idxs[i,:])
        target_pos[i] = torch.tensor((candi_idxs[i,:]==i).argmax(),dtype=torch.long)
        target_idx[i] = i
    
    shuffle_mask = random.sample(range(data_length),data_length)  
    
    candi_idxs = candi_idxs[shuffle_mask]
    target_pos = target_pos[shuffle_mask]
    target_idx = target_idx[shuffle_mask]
    
    candis_allx = []
    for i in range(candi_size):
        candis_allx.append(all_x[candi_idxs[:,i]])
    candis_allx = torch.stack(candis_allx)
    candis_allx = candis_allx.transpose(0,1)
    candis_tgtx = all_x[target_idx]
    return candis_tgtx, candis_allx, target_pos

def _gen_candis_valid(all_xt, all_xv, candi_size=15):
    import copy
    all_xtv = torch.cat((all_xv,all_xt),axis=0)
    total_length = all_xv.shape[0] + all_xt.shape[0]
    valid_length = all_xv.shape[0]
    candi_idxs = np.zeros((valid_length,candi_size))
    orign_smp_list = list(np.arange(0,total_length,1))
    target_pos = torch.zeros((valid_length,),dtype=torch.long)
    target_idx = torch.zeros((valid_length,),dtype=torch.long)
    
    for i in range(valid_length):
        tmp_smp_list = copy.deepcopy(orign_smp_list)
        tmp_smp_list.remove(i)   
        candi_idxs[i,0] = i
        candi_idxs[i,1:] = random.sample(tmp_smp_list,candi_size-1)  
        np.random.shuffle(candi_idxs[i,:])
        target_pos[i] = torch.tensor((candi_idxs[i,:]==i).argmax(),dtype=torch.long)
        target_idx[i] = i
    
    shuffle_mask = random.sample(range(valid_length),valid_length)  
    
    # ------ Shuffle the rows
    candi_idxs = candi_idxs[shuffle_mask]
    target_pos = target_pos[shuffle_mask]
    target_idx = target_idx[shuffle_mask]
    
    candis_allx = []
    for i in range(candi_size):
        candis_allx.append(all_xtv[candi_idxs[:,i]])
    candis_allx = torch.stack(candis_allx)
    candis_allx = candis_allx.transpose(0,1)
    candis_tgtx = all_xv[target_idx]
    return candis_tgtx, candis_allx, target_pos


def _get_allxyID(data_loader):
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
    all_x = torch.stack(all_x).reshape(-1, X_DIM).cuda()
    all_y = torch.stack(all_y).reshape(all_x.shape[0],-1).cuda()
    all_y = all_y.long()
    all_ID = torch.stack(all_ID).reshape(all_x.shape[0],-1).cuda()
    return (all_x, all_y, all_ID)

def _get_msg_types(agent, all_x):
    agent.eval()
    with torch.no_grad():
        _, msg = agent(all_x)
        gen_ID = msg[:,0]*NG1 +msg[:,1]
        gen_ID_list = gen_ID.tolist()
        gen_ID_set = set(gen_ID_list)
        return len(gen_ID_set)
 
def _get_valid_score_LSA_G12(agent, valid_loader):
    with torch.no_grad():
        agent.eval()
        tested_samples = 0
        correct_samples = 0
        tmp_loss = []
        for b_id, (x,y_origin,_) in enumerate(valid_loader):       
            x = x.cuda()
            y1_target = torch.tensor(y_origin[:,0],dtype=torch.long).cuda()
            y2_target = torch.tensor(y_origin[:,1],dtype=torch.long).cuda()
            y_hidden, msg = agent(x)
            msg = msg.cpu()
            #y_hidden = agent.clas_layer(hidden)
            y1_predict = msg[:,0]
            y2_predict = msg[:,1]
    
            loss =  CE_LOSS(y_hidden[:,:NG1],y1_target) + \
                    CE_LOSS(y_hidden[:,NG1:],y2_target)
            tmp_loss.append(loss.data.item())
            
            num_samples = y_origin.size(0)
            tested_samples += num_samples
            for i in range(y1_target.shape[0]):
                tmp_same = (y_origin[i]==msg[i]).sum()
                if tmp_same==2:
                    correct_samples += 1.                    
        accuracy = correct_samples/tested_samples
        loss_value = np.asarray(tmp_loss).mean()
        return accuracy, loss_value

       
def _get_valid_score_refgame(speaker, listener, valid_loader, candi_size):
    all_x_val, _, _ = _get_allxyID(valid_loader)  
    with torch.no_grad():
        speaker.eval()
        listener.eval()
        tested_samples = 0
        correct_samples = 0
        for i in range(20):
            candis_tgtx, candis_allx, target_pos = _gen_candis_valid(all_x, all_x_val, candi_size=candi_size)
            _, msg = speaker(candis_tgtx)
            pred_vector = listener(msg, candis_allx, gs_flag=False) 
            pred_pos = pred_vector.argmax(1).cpu()
            reward_mask = (target_pos-pred_pos)==0
            reward_mask = reward_mask.detach().int().cuda()
            tested_samples += float(reward_mask.shape[0])
            correct_samples += reward_mask.sum().float()
    return correct_samples/tested_samples

def _get_topsim_G12(agent, all_x):
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
    agent.eval()
    with torch.no_grad():
        data_len = all_x.shape[0]
        _, msg = agent(all_x)
        x = all_x.cpu()
        y = msg.cpu()
        xdist_pair, ydist_pair = [], []
        for i in range(data_len):
            for j in range(i):
                xdist_pair.append(_cal_dist(x[i,:],x[j,:],'cosine'))
                ydist_pair.append(_cal_dist(y[i,:],y[j,:],'hamming'))
        ydist_pair[-1] = 0.01 
        return pearsonr(xdist_pair,ydist_pair)[0]

PRE_ROUNDS = 1200
INT_ROUNDS = 4000

train_loader, valid_loader, zs_loader = Data_Gen_NonLinear_Map(batch_size=5, validation_split=0,random_seed=42, x_dim=X_DIM, samples=1, noise=0)
all_x, all_y, all_ID = _get_allxyID(train_loader)

"""
#### =================== Test the assisting functions ===
#pop_infa = new_pop_LSA(1)[0]
#data_xy = (all_x,all_y)
#pop_teen, pre_lang_results = pop_train_from_data_LSA_SGD(pop_infa, data_xy, all_x, PRE_ROUNDS, batch_size=64, see_valid=False, see_topsim=False, v_loader=train_loader)
#
#pop_infa = new_pop_LSA(1)[0]
#msg_types = _get_msg_types(pop_infa, all_x)
#topsim =  _get_topsim_G12(pop_infa, all_x)
#v_acc, v_loss = _get_valid_score_LSA_G12(pop_teen, train_loader)

## =================== TRIAL Interact and then generate data and the pretrain ===
pop_infa = new_pop_LSA(1)[0]
pop_adul, inter_results = pop_interact_refgame(pop_infa, all_x, INT_ROUNDS, batch_size=100, candi_size=15, see_valid=False,see_topsim=False, v_loader=train_loader)
# plt.plot(inter_results['spk_loss'])
data_xy = inter_results['data']
pop_infa = new_pop_LSA(1)[0]
pop_teen, pre_lang_results = pop_train_from_data_LSA_SGD(pop_infa, data_xy, all_x, PRE_ROUNDS, batch_size=7, see_valid=False, see_topsim=False, v_loader=train_loader)
## plt.plot(pre_lang_results['loss'])
valid_acc = _get_valid_score_refgame(pop_teen, valid_loader, candi_size=15)


### =================== TRIAL Interacting phase only ============
#pop_infa = new_pop_LSA(1)[0]
#pop_adul, inter_results = pop_interact_refgame(pop_infa, all_x, INT_ROUNDS, batch_size=100, candi_size=15, see_valid=False,see_topsim=False, v_loader=valid_loader)
## plt.plot(inter_results['spk_loss'])
##accuracy, loss_value = _get_valid_score_LSA_G12(pop_adul, valid_loader)      
######
#####
######pop_infa = new_pop_LSA(1)[0]
######pop_adul, inter_results = pop_interact_classification_LSA_SGD(pop_infa, train_loader, INT_ROUNDS, batch_size=16, see_valid=False, v_loader=valid_loader)
#####msg_types = _get_msg_types(pop_adul, all_x)


##
### =================== TRIAL one generation of NIL ============
#data_sampler = gen_samples(train_loader)
#xy_init = data_sampler.enumerate_xy_from_loader(rnd_flag=True)
#data_xy = xy_init
###==================== Infa learn from language becomes teenager ==========
#pop_infa = new_pop_LSA(1)[0]
#pop_adul, pre_lang_results = pop_train_from_data_LSA_SGD(pop_infa, data_xy, PRE_ROUNDS, batch_size=64, see_valid=True, see_topsim=True, v_loader=valid_loader)
## plt.plot(pre_lang_results['loss'])
#pop_infa2 = new_pop_LSA(1)[0]
#pop_teen, pre_lang_results2 = pop_train_from_agent_LSA_SGD(pop_infa2, pop_adul, all_x, 10000, batch_size=4,see_valid=True,see_topsim=True,v_loader=valid_loader)
## plt.plot(pre_lang_results2['loss'])
## ==================== Teenager interact becomes adults  ===========
#pop_adul, inter_results = pop_interact_rsa_LSA_SGD(pop_teen, all_x, all_ID, train_loader, INT_ROUNDS, batch_size=256, see_valid=True, v_loader=valid_loader)
## plt.plot(inter_results['loss'])
#accuracy, loss_value = _get_valid_score_LSA_G12(pop_adul, valid_loader)  
#####
##### ==================== Generate data from pop_adul ===========
####data = data_sampler.sample_xh_from_agent(pop_adul, 1000, logit_flag=False)


"""

# --------------- Learn from data version -----------------------------
for runs in range(1):
    valid_acc = []
    topsims = []
    diff_ID_size = []
    pop_infa = new_pop_LSA(1)[0]
    pop_adul, inter_results = pop_interact_refgame(pop_infa, all_x, INT_ROUNDS, batch_size=100, candi_size=15, see_valid=False,see_topsim=False, v_loader=zs_loader)
    data_xh = inter_results['data']
    
    for g in range(50):
        print(str(g)+"-",end="")      
        # ==================== Initial a new agent ============================
        pop_infa = new_pop_LSA(1)[0]
        # ==================== Infa learn from language becomes teenager ======
        pop_teen, pre_lang_results = pop_train_from_data_LSA_SGD(pop_infa, data_xh, all_x, PRE_ROUNDS, batch_size=4, see_valid=False, v_loader=zs_loader)
        # plt.plot(pre_lang_results['loss'])
        # ==================== Teenager interact becomes adults  ===========
        pop_adul, inter_results = pop_interact_refgame(pop_teen, all_x, INT_ROUNDS, batch_size=100, candi_size=15, see_valid=False,see_topsim=False, v_loader=zs_loader)
        # plt.plot(inter_results['rewards'])
        # ==================== Generate data from pop_adul ===========
        data_xh = inter_results['data']
        
        tmp_topsim = _get_topsim_G12(pop_adul, all_x)
        accuracy = inter_results['v_score']
        msg_types = _get_msg_types(pop_adul, all_x)   
        diff_ID_size.append(msg_types)
        valid_acc.append(accuracy)       
        topsims.append(tmp_topsim)
        
        print('topsim is %.3f, acc is %.3f, msg_types is %d'%(tmp_topsim,accuracy,msg_types))


## ================= See the topsim of perfect language ======================

#def _cal_dist(v1, v2, dis_type='hamming'):
#    '''
#        v1 and v2 should be two vectors with same length
#        dis_type should be `hamming` or `cosine`
#    '''
#    v1 = v1.reshape(1,-1)
#    v2 = v2.reshape(1,-1)
#    if dis_type.lower()=='hamming':
#        return torch.cdist(v1.float(),v2.float(),p=0)[0]
#    elif dis_type.lower()=='cosine':
#        return -nn.functional.cosine_similarity(v1,v2, dim=1)
#    elif dis_type.lower()=='eu':
#        pdist = nn.PairwiseDistance(p=2)
#        return pdist(v1,v2)
#    else:
#        print('dis_type should be `hamming` or `cosine` or `eu`')  
#
#
#x,y = all_x.cpu(),all_y.cpu()
#data_len = all_x.shape[0]
#xdist_pair = []
#ydist_pair = []
#for i in range(data_len):
#    for j in range(i):
#        xdist_pair.append(_cal_dist(x[i,:],x[j,:],'cosine'))
#        ydist_pair.append(_cal_dist(y[i,:],y[j,:],'hamming'))
#ydist_pair[-1] = 0.01 
#pearsonr(xdist_pair,ydist_pair)[0]