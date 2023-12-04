import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

colors = ['blue', 'red', 'orange','green']

original_his = 0.99
original_up = 0.7225
original_holdout = 0.3012
original_time = 0

base_dir = 'logs/5000'
settings = ['T5Small_r1e3b4', 'T5Small_r2e3b4', 'T5Small_r4e3b4']

r_list_small = []

for index, setting in enumerate(settings):
    with open(f'{base_dir}/{setting}/log.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        r_list_small.append(loaded_dict)


settings = ['T5Large_r1e3b4', 'T5Large_r2e3b4', 'T5Large_r4e3b4']
r_list_large = []

for index, setting in enumerate(settings):
    with open(f'{base_dir}/{setting}/log.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        r_list_large.append(loaded_dict)


t = list(range(0,5001,200))


HOLDOUT_SMALL = []
HOLDOUT_LARGE = []



for index, setting in enumerate(r_list_small):
    holdout_log = setting['all_HOLDOUT']
    holdout_f1 = [original_holdout]
    for x in t[1:]:
        holdout_f1.append(holdout_log[x]['holdout_f1'])
    HOLDOUT_SMALL.append(holdout_f1)


for index, setting in enumerate(r_list_large):
    holdout_log = setting['all_HOLDOUT']
    holdout_f1 = [original_holdout]
    for x in t[1:]:
        holdout_f1.append(holdout_log[x]['holdout_f1'])
    HOLDOUT_LARGE.append(holdout_f1)


fig, axs = plt.subplots(1, 2, figsize=(4,2),dpi=800)
for i in range(2):
    axs[i].set_box_aspect(0.75)



axs[0].plot(t, HOLDOUT_SMALL[0], label = '1 rank(s)/block',linewidth=1.8,color=colors[0])
axs[0].plot(t, HOLDOUT_SMALL[1], label = '2 rank(s)/block',linewidth=1.8,color=colors[1])
axs[0].plot(t, HOLDOUT_SMALL[2], label = '4 rank(s)/block',linewidth=1.8,color=colors[2])
axs[0].set_xlim(-200,5200)
axs[0].set_ylim(0.2,1.02)
axs[0].xaxis.set_major_locator(MultipleLocator(1000))
axs[0].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
axs[0].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[0].set_ylabel('Generality',fontweight='bold',family = 'serif')
axs[0].set_title('T5Small',fontweight='bold',family = 'serif',size=11)
axs[0].grid(True)


axs[1].plot(t, HOLDOUT_LARGE[0], label = 'PR = 1',linewidth=1.8,color=colors[0])
axs[1].plot(t, HOLDOUT_LARGE[1], label = 'PR = 2',linewidth=1.8,color=colors[1])
axs[1].plot(t, HOLDOUT_LARGE[2], label = 'PR = 4',linewidth=1.8,color=colors[2])
axs[1].set_xlim(-200,5200)
axs[1].set_ylim(0.2,1.02)
axs[1].xaxis.set_major_locator(MultipleLocator(1000))
axs[1].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[1].yaxis.set_major_locator(MultipleLocator(0.2))
axs[1].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[1].set_ylabel('Generality',fontweight='bold',family = 'serif')
axs[1].set_title('T5Large',fontweight='bold',family = 'serif',size=11)
axs[1].grid(True)




lines_labels = [axs[1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='lower center', ncol=3)
plt.tight_layout()
plt.subplots_adjust(wspace=0.6)
plt.subplots_adjust(wspace=0.6)
plt.subplots_adjust(bottom=0.35)

plt.savefig('plotres/T5Small_zsre_generality_42.jpg')

plt.show()

pass


