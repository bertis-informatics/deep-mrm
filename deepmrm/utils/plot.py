from matplotlib import pyplot as plt
import numpy as np


def plot_heavy_light_pair(time, xic, manual_bd=None, pred_bd=None):

    fig, axs = plt.subplots(2, sharex=True)
    for i in range(xic.shape[0]):
        
        axs[i].plot(time, ((-1)**(i+2))*xic[i, :, :].T)
        # axs[i].set_xticks([])
        axs[i].set_yticks([])
        # axs[i].axis("off")
        for bd, c, ls in zip([manual_bd, pred_bd], ['black', 'red'], ['dotted', 'dashed']):
            if bd is not None:
                axs[i].axvline(x=bd[0], linestyle=ls, color=c)
                axs[i].axvline(x=bd[1], linestyle=ls, color=c)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axs