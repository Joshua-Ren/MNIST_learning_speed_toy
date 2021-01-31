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
from funcs.language_pops import *
from funcs.language_figures import *



num_generations = 100
init_data = (['02','03','12','13'],
             ['aa','ab','bb','ba'])
data = init_data


results = []
topsim0_list = []
topsim1_list = []
for i in range(5):
    data = init_data
    result_run = []
    topsim0, topsim1 = [],[]
    for g in range(num_generations):
        pops_infa = new_pop(2)
        pops_teen = pop_train(pops_infa, data, 20)
        pops_adul, data = pop_interact_refgame(pops_teen,200)
        _ = pop_transmission(pops_adul,20, False)
        result_run.append(language_stats(pops_adul))
        topsim0.append(expect_topsim(pops_adul[0],topsims))
        topsim1.append(expect_topsim(pops_adul[1],topsims))       
    results.append(result_run)
    topsim0_list.append(topsim0)
    topsim1_list.append(topsim1)
    print(i)
plt.figure(1)
plot_ratio_result_graph(results)
plt.figure(2)
plt.plot(np.asarray(topsim0_list).mean(0))


