import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

colors = ['blue', 'red', 'orange','green']
base_dir = '../logs/VecDB'
settings = ['T5Small_5000_e50', 'T5Small_5000_e75', 'T5Small_5000_e100']
CLUSTER = []
CONFLICTS = []
FORGET = []

r_list = []

t = list(range(0,5001,200))

for index, setting in enumerate(settings):
    with open(f'{base_dir}/{setting}/log.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        r_list.append(loaded_dict)


for index, setting in enumerate(r_list):
    VecDB_log = setting['all_VecDB']
    cluster = [0]
    for x in t[1:]:
        cluster.append(VecDB_log[x]['num_cluster'])
    CLUSTER.append(cluster)



fig, axs = plt.subplots(1, 1, figsize=(4,4),dpi=800)
# axs = fig.axes()

axs.plot(t, CLUSTER[0], label = '$R$ = 50',linewidth=4,color=colors[0])
axs.plot(t, CLUSTER[1], label = '$R$ = 75',linewidth=4,color=colors[1])
axs.plot(t, CLUSTER[2], label = '$R$ = 100',linewidth=4,color=colors[2])
plt.axhline(y = 907, color = 'green', linestyle = '--',linewidth = 4,label = 'Answer Num.')
axs.set_xlim(-100,5100)
axs.set_ylim(0,4200)
axs.xaxis.set_major_locator(MultipleLocator(1000))
axs.set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs.yaxis.set_major_locator(MultipleLocator(1000))
axs.set_yticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs.set_xlabel('Edits',fontweight='bold',family = 'serif',fontsize=18)
axs.set_ylabel('Cluster Num',fontweight='bold',family = 'serif',fontsize=18)
axs.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(handlelength=2,fontsize=13)
plt.tight_layout()
plt.savefig("./ablation_res/T5Small_cluster")