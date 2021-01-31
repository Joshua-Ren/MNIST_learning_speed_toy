#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:49:35 2020

@author: joshua
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')
from math import log, log1p, exp
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy

from funcs.language_config import *
from funcs.language_funcs import *
from funcs.language_figures import *
from funcs.data_gen import Data_gen_toy_NIL
from funcs.models import MLP


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data 
import torch.optim as optim


CE_LOSS = nn.CrossEntropyLoss()
MSE_LOSS = nn.MSELoss()
SFTMX0 = nn.Softmax(0)


def cal_posterior_NIL(agent, data_loader):   
    
    def _hidden_to_logprob(hidden):
        epsi = 1e-8
        SFTMX = nn.Softmax(1)
        hidden_logits = hidden
        
        p1 = torch.log(SFTMX(hidden_logits[:,:2])*(1-epsi)+epsi)
        p2 = torch.log(SFTMX(hidden_logits[:,2:])*(1-epsi)+epsi)
        return torch.cat((p1,p2),axis=1)
    
    agent.eval()
    lang_logprob_table = []
    for lang_idx in range(256):
        language = mappings[lang_idx]
        statis_prob = {'02':[],'03':[],'12':[],'13':[]}
        for x,y,x_clas in data_loader:
            x = x.cuda()
            hidden = agent(x).detach()
            hid_prob = _hidden_to_logprob(hidden)
            # ----- for 02 ------     
            for obj, msg in language:
                msghot = MESSAGE_TO_LABEL_HOTS[msg]
                objclas = OBJECTS_TO_CLAS[obj]
                mask = (x_clas==objclas).squeeze()
                pos1, pos2 = msghot[0].argmax(), msghot[1].argmax()+2  
                statis_prob[obj].extend((hid_prob[mask][:,pos1] + hid_prob[mask][:,pos2]).tolist())
        lang_logprob = np.mean(statis_prob['02'])+np.mean(statis_prob['03'])+np.mean(statis_prob['12'])+np.mean(statis_prob['13'])
        #len(statis_prob['02']+statis_prob['03']+statis_prob['12']+statis_prob['13'])
        lang_logprob_table.append(lang_logprob)    
    agent.train()
    return np.asarray(lang_logprob_table)

def _NIL_sample_language(agent,xxc_4types):
    x_4types = xxc_4types[0]
    xc_4types = xxc_4types[1]
    smp_language = []
    for i in range(4):
        obj = OBJECTS[xc_4types[i]]
        hidden = agent(x_4types[i].cuda())
        smp_msg_dim1 = torch.bernoulli(SFTMX0(hidden[:2])[1]).cpu().detach().long()
        smp_msg_dim2 = torch.bernoulli(SFTMX0(hidden[2:])[1]).cpu().detach().long()
        msg = MESSAGES[2*smp_msg_dim1+smp_msg_dim2]
        smp_language.append([obj,msg])
    return smp_language

def _get_xxc_4types(data_loader):
    x_4types = []
    xc_4types = []
    TMP_LIST = [0,1,2,3]
    for x, y, xc in data_loader:
        for i in range(xc.shape[0]):
            if xc[i].item() in TMP_LIST:
                x_4types.append(x[i])
                xc_4types.append(xc[i])
                TMP_LIST.remove(xc[i].item())
                if len(TMP_LIST)==0:
                    return (x_4types, xc_4types)

def _sample_data_from_loader(data_loader):
    x, y, x_clas = [], [], [] 
    for tp_x, tp_y, tp_x_clas in data_loader:
        x.append(tp_x)
        y.append(tp_y)
        x_clas.append(tp_x_clas)
    x = torch.stack(x).reshape(-1,X_DIM).cuda()
    y = torch.stack(y).reshape(-1,2).cuda()
    x_clas = torch.stack(x_clas).reshape(-1,1)     
    return (x,y)

def _sample_data_from_agent(agent, data_loader, rsa_flag=False):
    x, y, x_clas = [], [], [] 
    for tp_x, tp_y, tp_x_clas in data_loader:
        x.append(tp_x)
        y.append(tp_y)
        x_clas.append(tp_x_clas)
    x = torch.stack(x).reshape(-1,X_DIM).cuda()
    y = torch.stack(y).reshape(-1,2).cuda()
    x_clas = torch.stack(x_clas).reshape(-1,1)      
     
    def _gen_rsay(language):    
        rsa_y = []
        for i in range(x.shape[0]):
            obj = OBJECTS[x_clas[i]]
            msg = msg_given_obj(language, obj, rsa_flag)
            rsa_y.append(MESSAGE_TO_LABEL_CLAS[msg])
        rsa_y = torch.stack(rsa_y).cuda()
        return rsa_y
    language = _NIL_sample_language(agent,xxc_4types)
    rsa_y = _gen_rsay(language)
    data = (x,rsa_y)
    return data


def new_pop_NIL(popsize):
    """
        We use probability distribution (256-length vector) to represent an agent
    """
    population = []
    for i in range(popsize):
        pop = MLP(in_dim=X_DIM, out_dim=OUT_DIM).cuda()
        population.append(pop)
    return population  


def pop_interact_rsa_NIL_SGD(agent, rounds, data_loader, xxc_4types): 
    results = {'loss':[],
               'accuracy':[]}
    agent.train()
    optim_agent = optim.Adam(agent.parameters(),lr=5e-4)
    rnd_idx = 0
    x, y, x_clas = [], [], [] 
    for tp_x, tp_y, tp_x_clas in data_loader:
        x.append(tp_x)
        y.append(tp_y)
        x_clas.append(tp_x_clas)
    x = torch.stack(x).reshape(-1,X_DIM).cuda()
    y = torch.stack(y).reshape(-1,2).cuda()
    x_clas = torch.stack(x_clas).reshape(-1,1)      
 
    def _gen_rsay(language):    
        rsa_y = []
        for i in range(x.shape[0]):
            obj = OBJECTS[x_clas[i]]
            msg = msg_given_obj(language, obj, True)
            rsa_y.append(MESSAGE_TO_LABEL_CLAS[msg])
        rsa_y = torch.stack(rsa_y).cuda()
        return rsa_y

    while (rnd_idx<rounds):
        if rnd_idx % int(rounds*0.5)==0:
            language = _NIL_sample_language(agent,xxc_4types)
            rsa_y = _gen_rsay(language)
        rnd_idx += 1
        rnd_mask = np.random.randint(0,x.shape[0],(4,))
        hidden = agent(x[rnd_mask])
        loss = CE_LOSS(hidden[:,:2],rsa_y[rnd_mask,0]) + CE_LOSS(hidden[:,2:],rsa_y[rnd_mask,1])  
        optim_agent.zero_grad()
        loss.backward()
        optim_agent.step() 
        results['loss'].append(loss.data.item())              
    return agent, results


def pop_train_from_data_NIL_SGD(agent, data, rounds):
    results = {'loss':[]}
    x = data[0].cuda()
    y = data[1].cuda()
    agent.train()
    optim_agent = optim.Adam(agent.parameters(),lr=5e-4)
    data_length = data[0].shape[0]
    rnd_idx = 0
    while (rnd_idx<rounds):
        rnd_idx += 1
        data_idx = np.random.randint(0,data_length,(1,))
        y_hidden = agent(x[data_idx])
        loss = CE_LOSS(y_hidden[0,:2].unsqueeze(0),y[data_idx,0]) + \
                CE_LOSS(y_hidden[0,2:].unsqueeze(0),y[data_idx,1])            
        optim_agent.zero_grad()
        loss.backward()
        optim_agent.step() 
        results['loss'].append(loss.data.item())  
    return agent, results


train_loader, _ = Data_gen_toy_NIL(mappings[89],x_dim=X_DIM, batch_size=32, validation_split=0., non_linear=True,noise=0.,samples=50)
xxc_4types = _get_xxc_4types(train_loader)

PRE_ROUNDS = 100
INT_ROUNDS = 500



data_init = _sample_data_from_loader(train_loader)
data = data_init

# ==================== Infa learn from language becomes teenager ==========
pop_infa = new_pop_NIL(1)[0]
pop_teen, pre_lang_results = pop_train_from_data_NIL_SGD(pop_infa, data, PRE_ROUNDS)
# plt.plot(pre_lang_results['loss'])
tmp_post_teen = cal_posterior_NIL(pop_teen, train_loader)
draw_probs(tmp_post_teen,1e-5,'Teen, learn from language',log=True) 

# ==================== Teenager interact becomes adults  ===========
pop_adul, inter_results = pop_interact_rsa_NIL_SGD(pop_teen, INT_ROUNDS, train_loader, xxc_4types)
#plt.plot(inter_results['loss'])
tmp_post_adul = cal_posterior_NIL(pop_adul, train_loader)
draw_probs(tmp_post_adul,1e-5,'Teen, learn from language',log=True) 

# ==================== Generate data from pop_adul ===========
data = _sample_data_from_agent(pop_adul, train_loader, rsa_flag=True)

"""
results_run = []
results_pre = []
accuracys = []
topsim0_list = []
for i in range(1):
    result_run = []
    result_pre = []
    topsim0 = []

    data_init = _sample_data_from_loader(train_loader)
    data = data_init
    for g in range(20):
        print(str(g)+"-",end="")      
        # ==================== Initial a new agent ============================
        pop_infa = new_pop_NIL(1)[0]
        # ==================== Infa learn from language becomes teenager ======
        pop_teen, pre_lang_results = pop_train_from_data_NIL_SGD(pop_infa, data, PRE_ROUNDS) 
        # plt.plot(pre_lang_results['loss'])
        # ==================== Teenager interact becomes adults  ===========
        pop_adul, inter_results = pop_interact_rsa_NIL_SGD(pop_teen,INT_ROUNDS, train_loader,xxc_4types)
        # plt.plot(inter_results['loss'])
        # ==================== Generate data from pop_adul ===========
        data = _sample_data_from_agent(pop_adul, train_loader, rsa_flag=True)
      
#        tmp_post_teen = cal_posterior_NIL(pop_teen, train_loader)
#        result_pre.append(language_stats([tmp_post_teen]))      
        tmp_post_adul = cal_posterior_NIL(pop_adul, train_loader) # draw_probs(tmp_post) to see distribution              
        result_run.append(language_stats([tmp_post_adul]))
        topsim0.append(expect_topsim(tmp_post_adul,topsims))        
    results_pre.append(result_pre)
    results_run.append(result_run)
    topsim0_list.append(topsim0)
#draw_probs(tmp_post_adul,1e-5,'Teen, learn from language',log=True)     
plt.figure(1)
plot_ratio_result_graph(results_run)
plt.figure(2)
plt.plot(topsim0)
#plt.figure(3)
#plt.plot(accuracys)
#plt.plot(inter_results['loss'])

"""






"""
# ====================Other Learning phase designs ============================
def pop_train_from_teacher_NIL_SGD(agent1, agent2, rounds, data_loader, mse_or_ce='mse'):
    agent1.train()      # Student
    optim_student = optim.Adam(agent1.parameters(),lr=5e-4)
    rnd_idx = 0
    results = {'loss':[]}
    
    x, y, x_clas = [], [], []
    for tp_x, tp_y, tp_x_clas in data_loader:
        x.append(tp_x)
        y.append(tp_y)
        x_clas.append(tp_x_clas)
    x = torch.stack(x).reshape(-1,X_DIM).cuda()
    y = torch.stack(y).reshape(-1,2).cuda()
    x_clas = torch.stack(x_clas).reshape(-1,1)            
    
    while (rnd_idx<rounds):
        rnd_idx += 1
        data_idx = np.random.randint(0,x.shape[0],(1,))
        h_teach = agent2(x[data_idx]).detach()
        h_study = agent1(x[data_idx])

        if mse_or_ce.lower() == 'mse':
            loss = MSE_LOSS(h_study,h_teach)
        elif mse_or_ce.lower() == 'ce':
            tgt1 = h_teach[:,:2].argmax(1).long().cuda()
            tgt2 = h_teach[:,2:].argmax(1).long().cuda()
            loss = CE_LOSS(h_study[:,:2],tgt1) + CE_LOSS(h_study[:,2:],tgt2)        

        optim_student.zero_grad()
        loss.backward()
        optim_student.step() 
        results['loss'].append(loss.data.item())                 
    return agent1, results

# ====================Other Interact phase designs ============================
def pop_interact_diff_NIL(agent, rounds, data_loader):
    '''
        Interacting phase using diff-rand-select method.
        Each time we give a pair (x1,x2) to agent, the agent will generate
        (y1,y2), then:
            if x1==x2:
                if y1==y2: correct, use <x1,y1> to update params.
                if y1!=y2: incorrect, use <x1,y1> and <x2,y1> to update params.
                           [only update params of h2?]
            if x1!=x2:
                if y1!=y2: correct,use <x1,y1> and <x2,y2> to update params.
                if y1==y2: incorrect, use <x1,y1> and <x2,rand(y2)> to update
                
    '''
    def gen_rand_notsame(y):
        rand_y = torch.tensor(np.random.randint(0,2,(2,))).cuda()
        while (rand_y == y).sum()==2:
            rand_y = torch.tensor(np.random.randint(0,2,(2,))).cuda()  
        return rand_y

    half_batch = int(data_loader.batch_size*0.5)
    results = {'loss':[],
               'accuracy':[]}
    agent.train()
    optim_agent = optim.Adam(agent.parameters(),lr=5e-4)
    rnd_idx = 0
    while (rnd_idx<rounds):
        for x, y, x_clas in data_loader:
            rnd_idx += 1
            x1, x2 = x[:half_batch,:].cuda(), x[half_batch:,:].cuda()
            c1, c2 = x_clas[:half_batch,:].cuda(), x_clas[half_batch:,:].cuda()
            tf_target = (c1==c2).long()
            
            h1 = agent(x1)
            y1_hat = torch.cat((h1[:,:2].argmax(1).unsqueeze(1), h1[:,2:].argmax(1).unsqueeze(1)),axis=1)
            h2 = agent(x2)
            y2_hat = torch.cat((h2[:,:2].argmax(1).unsqueeze(1), h2[:,2:].argmax(1).unsqueeze(1)),axis=1)
            tf_predict = ((y1_hat==y2_hat).sum(1)==2).unsqueeze(1).long()
            
            x_inter = []
            y_inter = []
            for b_idx in range(half_batch):
                tgt_pred_flag = torch.cat((tf_target,tf_predict),axis=1)[b_idx]     
                BOOL_0 = torch.tensor([0]).cuda()   # BOOL_0 is DIFFERENT
                BOOL_1 = torch.tensor([1]).cuda()   # BOOL_1 is SAME
                if (tgt_pred_flag[0] == BOOL_0 and tgt_pred_flag[1] == BOOL_0):
                    a=0
#                    x_inter.append(x1[b_idx,:])
#                    x_inter.append(x2[b_idx,:])
#                    y_inter.append(y1_hat[b_idx])
#                    y_inter.append(y2_hat[b_idx])
                elif (tgt_pred_flag[0] == BOOL_1 and tgt_pred_flag[1] == BOOL_1):
                    a=0
#                    x_inter.append(x1[b_idx,:])
#                    y_inter.append(y1_hat[b_idx])
                elif (tgt_pred_flag[0] == BOOL_1 and tgt_pred_flag[1] == BOOL_0):
                    a=0
                    x_inter.append(x1[b_idx,:])
                    x_inter.append(x2[b_idx,:])
                    y_inter.append(y1_hat[b_idx])
                    y_inter.append(y1_hat[b_idx])                    
                elif (tgt_pred_flag[0] == BOOL_0 and tgt_pred_flag[1] == BOOL_1):       # random gen y2 when x1!=x2, y1=y2
                    x_inter.append(x1[b_idx,:])
                    x_inter.append(x2[b_idx,:])
                    y_inter.append(y1_hat[b_idx])
                    rand_y2 = gen_rand_notsame(y1_hat[b_idx])
                    y_inter.append(rand_y2)
            if len(x_inter)>0:
                x_inter_batch = torch.stack(x_inter)
                y_inter_batch = torch.stack(y_inter)
                
                y_hidden = agent(x_inter_batch)
                loss = CE_LOSS(y_hidden[:,:2],y_inter_batch[:,0]) + CE_LOSS(y_hidden[:,2:],y_inter_batch[:,1])            
                optim_agent.zero_grad()
                loss.backward()
                optim_agent.step() 
                results['loss'].append(loss.data.item())  
            
    return agent, results     
         
            
def pop_interact_tf_NIL(agent, rounds, data_loader):
    '''
        Interacting phase using true or false design.
        We split the batch data to (x1,x2) then feed x1, x2 to the agent.
        The agent will give h1, h2. We feed h1, h2 to the tf_layer of the agent,
        and then predict whether clas(x1)=clas(x2)
    '''
    half_batch = int(data_loader.batch_size*0.5)
    results = {'loss':[],
               'accuracy':[]}
    agent.train()
    optim_agent = optim.Adam(agent.parameters(),lr=5e-4)
    optim_agent_tflayer = optim.Adam(agent.tf_layer.parameters(),lr=5e-4)
    rnd_idx = 0
    while (rnd_idx<rounds):
        for x,y,x_clas in data_loader:
            rnd_idx += 1
            x1, x2 = x[:half_batch,:].cuda(), x[half_batch:,:].cuda()
            c1, c2 = x_clas[:half_batch,:].cuda(), x_clas[half_batch:,:].cuda()
            tf_target = (c1==c2).long().reshape(half_batch,).cuda()
            h1 = agent(x1)
            h2 = agent(x2).detach()
            #h1_cat_h2 = torch.cat((h1.unsqueeze(1),h2.unsqueeze(1)),axis=1)
            h1_cat_h2 = torch.cat((h1,h2),axis=1)
            hidden = agent.tf_layer(h1_cat_h2)
            
            tf_predict = (hidden.argmax(1))
            results['accuracy'].append((tf_predict== tf_target).sum().data.item())
            
            loss = CE_LOSS(hidden,tf_target)
            if rnd_idx < int(INT_ROUNDS*0.6):
                optim_agent_tflayer.zero_grad()
                loss.backward()
                optim_agent_tflayer.step() 
                results['loss'].append(loss.data.item())                  
            else:
                optim_agent.zero_grad()
                loss.backward()
                optim_agent.step() 
                results['loss'].append(loss.data.item())    
    return agent, results

def pop_interact_classification_NIL(agent, rounds, data_loader):
    '''
        Let agent generate message for each input, then use the message to do 
        classification task.
    '''
    results = {'loss':[],
               'accuracy':[]}
    agent.train()
    optim_agent = optim.Adam(agent.parameters(),lr=5e-4)
    optim_agent_claslayer = optim.Adam(agent.clas_layer.parameters(),lr=5e-4)
    rnd_idx = 0
    while (rnd_idx<rounds):
        for x,y,x_clas in data_loader:
            rnd_idx += 1
            x = x.cuda()
            x_clas = x_clas.squeeze().long().cuda()
            hidden = agent(x)
            hidden_clas = agent.clas_layer(hidden)
            loss = CE_LOSS(hidden_clas, x_clas)
            
            if rnd_idx < int(INT_ROUNDS*0.2):
                optim_agent_claslayer.zero_grad()
                loss.backward()
                optim_agent_claslayer.step() 
                results['loss'].append(loss.data.item())                  
            else:
                optim_agent.zero_grad()
                loss.backward()
                optim_agent.step() 
                results['loss'].append(loss.data.item())  
    return agent, results
"""










