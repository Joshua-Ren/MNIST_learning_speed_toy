#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:59:41 2020

@author: joshua
"""
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from .configs import *
from scipy.stats import pearsonr, spearmanr

class permut_g12():
    '''
    We need to transfer (g1,g2) to (g1',g2')
    Use the following to test:
        g1, g2 = 1,2
        permuter = permut_g12(ratio_perm=0.5)
        gg1,gg2 = permuter.g123_to_gg123(g1, g2)
    '''
    def __init__(self, ratio_perm):
        super(permut_g12, self).__init__()
        self.ratio_perm = ratio_perm
        self.total_size = NG1*NG2
        self.M = self.gen_perm_M()
    
    def gen_perm_M(self):
        # === The first dim of M is origin, the second is mapped====
        num_perm = np.int(self.ratio_perm*self.total_size)
    
        M = np.zeros((self.total_size,self.total_size))
        origin_mask = np.arange(0,self.total_size,1)
        permute_mask = np.arange(0,self.total_size,1)
        select_mask = random.sample(range(0,self.total_size), num_perm)
        select_copy = copy.deepcopy(select_mask)
        np.random.shuffle(select_copy)
        permute_mask[select_mask] = select_copy
        
        M[origin_mask, permute_mask] = 1
        
        return M
    
    def idx_to_g12(self,idx):
        gg2 = np.int(idx/NG1)
        gg1 = np.int(idx-NG1*gg2)
        return gg1, gg2
    
    def g12_to_idx(self,g1,g2):
        idx = g1 + NG1*g2
        return idx
    
    def get_idx_perm(self,M,idx):
        return np.where(M[idx]==1)[0][0]
    
    def g12_to_gg12(self, g1,g2):
        idx = self.g123_to_idx(g1,g2)
        idx_perm = self.get_idx_perm(self.M,idx)
        gg1, gg2, = self.idx_to_g12(idx_perm)
        return gg1,gg2



