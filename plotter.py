from cProfile import label
from shutil import which
from turtle import color
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import pandas as pd
import datetime
import numpy as np
plot_dict = {'info_value': 'Information value', 'waiting_time': 'Waiting time (s)'}
legend = True
colors = ['xkcd:light red', 'xkcd:pumpkin',  'xkcd:dark magenta', 
        'xkcd:lavender', 'xkcd:burple', 'xkcd:light magenta',
        'xkcd:jade', 'xkcd:neon green', 'xkcd:dull yellow',
        'xkcd:aqua blue', 'xkcd:blue grey', 'xkcd:black']
for plot_value in plot_dict.keys():
    diameter, trigger, prep, bandw = [500, 1000, 2000, 4000], [1000, 10000, 100000], [0.1], [10, 100, 1000]
    
    if legend:
        fig, ax = plt.subplots(figsize=(20,20))
    else:
        fig, ax = plt.subplots()
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)

    # colors = ['xkcd:light red', 'xkcd:dark magenta', 'xkcd:pastel pink', 'xkcd:pinkish', 'xkcd:pumpkin', 'xkcd:orange yellow', 
    # 'xkcd:light blue', 'xkcd:bright blue', 'xkcd:lavender', 'xkcd:neon purple', 'xkcd:blue grey', 'xkcd:deep blue',
    # 'xkcd:light blue green', 'xkcd:jade', 'xkcd:jungle green', 'xkcd:puke yellow', 'xkcd:pale brown', 'xkcd:shit']
    # diameter, trigger
    handles = []
    for trigger_idx, trigger_i in enumerate(trigger):
        for prep_idx, prep_i in enumerate(prep):
            for bandw_idx, bandw_i in enumerate(bandw):            
                data = []
                for diameter_idx, diameter_i  in enumerate(diameter):
                    result_df = pd.read_csv(f'./new_simulation_results/result_wt{prep_i}d{diameter_i}tt{trigger_i}bw{bandw_i}.csv')
                    if plot_value == 'waiting_time':
                        data.append(result_df[plot_value].mean()/1000)
                    else:
                        data.append(result_df[plot_value].mean())
                p, = ax.plot(diameter, data, label=f'{trigger_i / 1000} | {bandw_i}', color=colors[sum([trigger_idx * 3, bandw_idx])], linewidth=3)
                handles.append(p)
    ax.set_xscale('log')
    ax.set_xticks(diameter, which='major')
    #ax.set_xticks([], which='minor')
    plt.minorticks_off()
    if plot_value == 'waiting_time':
        ax.set_yscale('log')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = mt.ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        if axis == ax.xaxis:
             axis.set_minor_formatter(formatter)
    ax.set_xlabel('Diameter [m]', fontsize=18)
    ax.set_ylabel(plot_dict[plot_value], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='x', which='minor', labelsize=15)
    p, = ax.plot([], [], ' ', label='$T_{t} | Bandwidth$')
    if legend:
        leg = ax.legend(handles=handles.append(p), bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=20)
        texts = leg.get_texts()
        last_text = texts[-1]
        last_text._fontproperties = texts[0]._fontproperties.copy()
        last_text.set_fontsize(20)
    fig.tight_layout()
    fig.savefig(f'./graphs/{plot_value}_diameter.pdf')

for plot_value in plot_dict.keys():
    diameter, trigger, prep, bandw = [500, 1000, 2000, 4000], [1000, 10000, 100000], [0.1], [10, 100, 1000]

    if legend:
        fig, ax = plt.subplots(figsize=(20,20))
    else:
        fig, ax = plt.subplots()
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)

    # colors = ['xkcd:light red', 'xkcd:dark magenta', 'xkcd:pastel pink', 'xkcd:pinkish', 'xkcd:pumpkin', 'xkcd:orange yellow', 
    # 'xkcd:light blue', 'xkcd:bright blue', 'xkcd:lavender', 'xkcd:neon purple', 'xkcd:blue grey', 'xkcd:deep blue',
    # 'xkcd:light blue green', 'xkcd:jade', 'xkcd:jungle green', 'xkcd:puke yellow', 'xkcd:pale brown', 'xkcd:shit']
    # diameter, trigger
    handles = []
    for diameter_idx, diameter_i  in enumerate(diameter):
        for prep_idx, prep_i in enumerate(prep):
            for bandw_idx, bandw_i in enumerate(bandw):
                data = []
                for trigger_idx, trigger_i in enumerate(trigger):
                    result_df = pd.read_csv(f'./new_simulation_results/result_wt{prep_i}d{diameter_i}tt{trigger_i}bw{bandw_i}.csv')
                    if plot_value == 'waiting_time':
                        data.append(result_df[plot_value].mean()/1000)
                    else:
                        data.append(result_df[plot_value].mean())
                p, = ax.plot(trigger, data, label=f'{diameter_i} | {bandw_i}', color=colors[sum([diameter_idx * 3, bandw_idx])], linewidth=3)
                handles.append(p)
    ax.set_xscale('log')
    ax.set_xticks(trigger, which='major')
    #ax.set_xticks([], which='minor')
    plt.minorticks_off()
    if plot_value == 'waiting_time':
        ax.set_yscale('log')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = mt.ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        if axis == ax.xaxis:
             axis.set_minor_formatter(formatter)
    ax.set_xlabel('Trigger time [s]', fontsize=18)
    ax.set_ylabel(plot_dict[plot_value], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='x', which='minor', labelsize=15)
    p, = ax.plot([], [], ' ', label='$Diameter | Bandwidth$')
    if legend:
        leg = ax.legend(handles=handles.append(p), bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=20)
        texts = leg.get_texts()
        last_text = texts[-1]
        last_text._fontproperties = texts[0]._fontproperties.copy()
        last_text.set_fontsize(20)
    fig.tight_layout()
    fig.savefig(f'./graphs/{plot_value}_trigger.pdf')

for plot_value in plot_dict.keys():
    diameter, trigger, prep, bandw = [500, 1000, 2000, 4000], [1000, 10000, 100000], [0.1], [10, 100, 1000]

    if legend:
        fig, ax = plt.subplots(figsize=(20,20))
    else:
        fig, ax = plt.subplots()
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)

    # colors = ['xkcd:light red', 'xkcd:dark magenta', 'xkcd:pastel pink', 'xkcd:pinkish', 'xkcd:pumpkin', 'xkcd:orange yellow', 
    # 'xkcd:light blue', 'xkcd:bright blue', 'xkcd:lavender', 'xkcd:neon purple', 'xkcd:blue grey', 'xkcd:deep blue',
    # 'xkcd:light blue green', 'xkcd:jade', 'xkcd:jungle green', 'xkcd:puke yellow', 'xkcd:pale brown', 'xkcd:shit']
    # diameter, trigger
    handles = []
    for trigger_idx, trigger_i in enumerate(trigger):
        for prep_idx, prep_i in enumerate(prep):
            for diameter_idx, diameter_i  in enumerate(diameter):
                data = []
                for bandw_idx, bandw_i in enumerate(bandw):
                    result_df = pd.read_csv(f'./new_simulation_results/result_wt{prep_i}d{diameter_i}tt{trigger_i}bw{bandw_i}.csv')
                    if plot_value == 'waiting_time':
                        data.append(result_df[plot_value].mean()/1000)
                    else:
                        data.append(result_df[plot_value].mean())
                p, = ax.plot(bandw, data, label=f'{trigger_i / 1000} | {diameter_i}', color=colors[sum([trigger_idx * 4, diameter_idx])], linewidth=3)
                handles.append(p)
    ax.set_xscale('log')
    ax.set_xticks(bandw, which='major')
    #ax.set_xticks([], which='minor')
    plt.minorticks_off()
    if plot_value == 'waiting_time':
        ax.set_yscale('log')
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = mt.ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        if axis == ax.xaxis:
             axis.set_minor_formatter(formatter)
    ax.set_xlabel('Bandwidth [Mb/s]', fontsize=18)
    ax.set_ylabel(plot_dict[plot_value], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='x', which='minor', labelsize=15)
    p, = ax.plot([], [], ' ', label='$T_{t} | Diameter$')
    if legend:
        leg = ax.legend(handles=handles.append(p), bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=20)
        texts = leg.get_texts()
        last_text = texts[-1]
        last_text._fontproperties = texts[0]._fontproperties.copy()
        last_text.set_fontsize(20)
    fig.tight_layout()
    fig.savefig(f'./graphs/{plot_value}_bw.pdf')

"""
#curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
p1, = ax.plot(result_df.index, result_df['response_time'], color='xkcd:red', label='Response time (ms)')
ax.set_xlabel('Aggregations') 
ax.set_ylabel('Response time (ms)', color='xkcd:red')
ax2 = ax.twinx()
p2, = ax2.plot(result_df.index, result_df['vehicle_density'], color='xkcd:grey', label=f'Vehicle density (vehicle/km\N{SUPERSCRIPT TWO})')
ax2.set_ylabel('Vehicle density (vehicle/km\N{SUPERSCRIPT TWO})', color='xkcd:grey')
ax3 = ax.twinx()
p3, = ax3.plot(result_df.index, result_df['info_value'], color='xkcd:blue', label='Information value')
ax3.set_ylabel('Information value', color='xkcd:blue')
ax.legend(handles=[p1, p2, p3], loc='best')
ax3.spines['right'].set_position(('outward', 80))
fig.tight_layout()
fig.savefig(f'./simulation_results/figure_{curr_time}.pdf')
"""