import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

colors = ['blue', 'red', 'green','orange']

original_his = 0.99
original_up = 0.8111
original_holdout = 0.3012
original_time = 0

base_dir = 'logs/5000'
settings = ['T5Large_r2e1b4', 'T5Large_r2e6b4', 'T5Large_r2e20b4']

r_list = []

for index, setting in enumerate(settings):
    with open(f'{base_dir}/{setting}/log.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        r_list.append(loaded_dict)


t = list(range(0,5001,200))

UP = []
HIS = []
HOLDOUT = []
TIME = []
for index, setting in enumerate(r_list):
    up_log = setting['all_UP']
    up_f1 = [original_up]
    for x in t[1:]:
        up_f1.append(up_log[x]['UP_f1'])
    UP.append(up_f1)

for index, setting in enumerate(r_list):
    his_log = setting['all_HIS']
    his_f1 = [his_log[200]['HIS_f1']]
    for x in t[1:]:
        his_f1.append(his_log[x]['HIS_f1'])
    HIS.append(his_f1)

for index, setting in enumerate(r_list):
    holdout_log = setting['all_HOLDOUT']
    holdout_f1 = [original_holdout]
    for x in t[1:]:
        holdout_f1.append(holdout_log[x]['holdout_f1'])
    HOLDOUT.append(holdout_f1)

for index, setting in enumerate(r_list):
    time_log = setting['all_edit_time']
    time_list = [0]
    for x in t[1:]:
        time_list.append(time_log[x]/60)
    TIME.append(time_list)

fig, axs = plt.subplots(1, 4, figsize=(8,2),dpi=800)
for i in range(4):
    axs[i].set_box_aspect(0.85)



axs[0].plot(t, HIS[0], label = '1 rank(s)/block',linewidth=1.8,color=colors[0])
axs[0].plot(t, HIS[1], label = '2 rank(s)/block',linewidth=1.8,color=colors[1])
axs[0].plot(t, HIS[2], label = '4 rank(s)/block',linewidth=1.8,color=colors[2])
axs[0].set_xlim(-200,5200)
axs[0].set_ylim(0.68,1.02)
axs[0].xaxis.set_major_locator(MultipleLocator(1000))
axs[0].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[0].yaxis.set_major_locator(MultipleLocator(0.1))
axs[0].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[0].set_ylabel('Edit Seccess',fontweight='bold',family = 'serif')
axs[0].grid(True)

axs[1].plot(t, HOLDOUT[0], label = '1 rank(s)/block',linewidth=1.8,color=colors[0])
axs[1].plot(t, HOLDOUT[1], label = '2 rank(s)/block',linewidth=1.8,color=colors[1])
axs[1].plot(t, HOLDOUT[2], label = '4 rank(s)/block',linewidth=1.8,color=colors[2])
axs[1].set_xlim(-200,5200)
axs[1].set_ylim(0.2,1.02)
axs[1].xaxis.set_major_locator(MultipleLocator(1000))
axs[1].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[1].yaxis.set_major_locator(MultipleLocator(0.2))
axs[1].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[1].set_ylabel('Holdout',fontweight='bold',family = 'serif')
axs[1].grid(True)

axs[2].plot(t, TIME[0], label = '1 rank(s)/block',linewidth=1.8,color=colors[0])
axs[2].plot(t, TIME[1], label = '2 rank(s)/block',linewidth=1.8,color=colors[1])
axs[2].plot(t, TIME[2], label = '4 rank(s)/block',linewidth=1.8,color=colors[2])
axs[2].set_xlim(-200,5200)
axs[2].set_ylim(0,15)
axs[2].xaxis.set_major_locator(MultipleLocator(1000))
axs[2].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[2].yaxis.set_major_locator(MultipleLocator(3))
axs[2].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[2].set_ylabel('Time (mins)',fontweight='bold',family = 'serif')
axs[2].grid(True)


axs[3].plot(t, UP[0], label = 'Radius = 1.0',linewidth=1.8,color=colors[0])
axs[3].plot(t, UP[1], label = 'Radius = 6.0',linewidth=1.8,color=colors[1])
axs[3].plot(t, UP[2], label = 'Radius = 20.0',linewidth=1.8,color=colors[2])
axs[3].set_xlim(-200,5200)
axs[3].set_ylim(0.5,0.9)
axs[3].xaxis.set_major_locator(MultipleLocator(1000))
axs[3].set_xticklabels(['0', '0','1k', '2k', '3k','4k','5k'])
axs[3].yaxis.set_major_locator(MultipleLocator(0.1))
axs[3].set_xlabel('Edits',fontweight='bold',family = 'serif')
axs[3].set_ylabel('Locality',fontweight='bold',family = 'serif')
axs[3].grid(True)


lines_labels = [axs[3].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='lower center', ncol=4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.4)


plt.savefig('plotres/T5Large_zsre_eps.jpg')

plt.show()

pass


