#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:46:32 2020

@author: joshua
"""
import numpy as np
SIGMOID_MODE = True
CLASSIFY_NUM = 2
NG1, NG2=10, 10
X_DIM = NG1+NG2
HID_SIZE = 128
MID_SIZE = 10
ZS_TABLE = []
#ZS_TABLE = [(9,9)]


ZS_RATIO = 0.5

for g1 in range(NG1):
    for g2 in range(NG2):
        if np.random.uniform(0,1,(1,)) < ZS_RATIO:
            ZS_TABLE.append((g1,g2))
#ZS_TABLE = [(0, 0), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 0), (1, 4), (1, 6), (1, 8), (1, 9), (2, 2), (2, 6), (2, 7), (2, 8), (3, 0), (3, 1), (3, 5), (3, 8), (4, 2), (4, 4), (4, 5), (4, 7), (4, 8), (4, 9), (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 7), (6, 0), (6, 1), (6, 3), (6, 4), (7, 1), (7, 2), (7, 4), (7, 5), (7, 7), (8, 3), (8, 4), (8, 6), (9, 0), (9, 2), (9, 4), (9, 5), (9, 8), (0, 0), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 0), (1, 2), (1, 4), (1, 6), (1, 8), (1, 9), (2, 2), (2, 6), (2, 7), (2, 8), (3, 0), (3, 1), (3, 5), (3, 8), (4, 2), (4, 4), (4, 5), (4, 7), (4, 8), (4, 9), (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 7), (6, 0), (6, 1), (6, 3), (6, 4), (7, 1), (7, 2), (7, 4), (7, 5), (7, 7), (8, 3), (8, 4), (8, 6), (9, 0), (9, 2), (9, 4), (9, 5), (9, 8)]



