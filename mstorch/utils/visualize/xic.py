import io

import PIL
import torch
import numpy as np
from matplotlib import pyplot as plt


color = ['r', 'g', 'b', 'pink', 'cyan', 'black', 'yellow']
peptide_prefix = ['light', 'heavy']

def create_peak_pair_image(
        time, 
        xic_tensor, 
        selected_trans_idx, 
        xlim=None,
        rt=None, 
        title_str=None):
    
    assert xic_tensor.shape[0] == 2, "Require a pair of heavy & light peptides"

    light_xic = xic_tensor[0, selected_trans_idx, :]
    heavy_xic = xic_tensor[1, selected_trans_idx, :]

    color = ['r', 'g', 'b', 'pink', 'cyan', 'black', 'yellow']
    fig, axs = plt.subplots(1)
    axs.set_title(title_str)
    axs.plot(time, light_xic, c='b', linewidth=1)
    axs.plot(time, heavy_xic, c='r', linewidth=2)
    axs.get_yaxis().set_visible(False)        
    if xlim:
        axs.set_xlim(xlim)
    else:
        axs.set_xlim([time[0], time[-1]])
    axs.legend([f'{s}-{selected_trans_idx+1}' for s in peptide_prefix])
    if rt:
        axs.axvline(x=rt, linestyle='dashed', color='gray')

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_img = PIL.Image.open(buf).convert('RGB')

    return pil_img


def create_peak_group_image(
                    time, 
                    xic_tensor, 
                    rt=None, 
                    boundary=None,
                    xlim=None,
                    title_str=None,
                    show_legend=True):

    assert xic_tensor.shape[0] == 2, "Require a pair of heavy & light peptides"
    fig, axs = plt.subplots(xic_tensor.shape[0], sharex=True)
    if title_str:
        axs[0].set_title(title_str)
    for i in range(xic_tensor.shape[0]):

        legend_str = [f'{peptide_prefix[i]}-{t+1}' for t in range(xic_tensor.shape[1])]

        # for each peptide (transition group)
        y = xic_tensor[i, :]
        # if isinstance(xic_tensor, torch.Tensor):
        #     y = y.numpy()
        for j in range(xic_tensor.shape[1]):
            # plot each transition XIC
            axs[i].plot(time, y[j]) #, c=color[j])
        axs[i].get_yaxis().set_visible(False)        
        if xlim:
            axs[i].set_xlim(xlim)
        else:        
            axs[i].set_xlim([time[0], time[-1]])
        if rt:
            axs[i].axvline(x=rt, linestyle='dashed', color='gray')

        if boundary:
            axs[i].axvline(x=boundary[0], linestyle='dashed', color='black')
            axs[i].axvline(x=boundary[1], linestyle='dashed', color='black')

        if show_legend:
            axs[i].legend(legend_str)
    
    #plt.savefig('temp.jpg', bbox_inches='tight', pad_inches=0)
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_img = PIL.Image.open(buf).convert('RGB')

    return pil_img


def create_transition_pair_image(
            time, xic, 
            transition_pairs=3, rt=None, 
            title_str=None, window_size=3):
    
    assert len(xic) % transition_pairs == 0, "Require a pair of heavy & light peptides"
    assert len(time) == len(xic)
    
    fig, axs = plt.subplots(2, sharex=True)
    if title_str:
        axs[0].set_title(title_str)
    for i in range(2):

        legend_str = [f'{peptide_prefix[i]}-{t+1}' for t in range(transition_pairs)]

        st = i*transition_pairs
        ed = st + transition_pairs
        for j in range(st, ed):
            # plot each transition XIC
            axs[i].plot(time[j], xic[j], c=color[j])
        axs[i].get_yaxis().set_visible(False)        
        if window_size and rt:
            axs[i].set_xlim([rt-window_size*0.5, rt+window_size*0.5])
        else:
            axs[i].set_xlim([time[0][0], time[0][-1]])
        if rt:
            axs[i].axvline(x=rt, linestyle='dashed', color='gray')
        axs[i].legend(legend_str)
    
    #plt.savefig('temp.jpg', bbox_inches='tight', pad_inches=0)
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    pil_img = PIL.Image.open(buf).convert('RGB')

    return pil_img
