from matplotlib import pyplot as plt
import numpy as np


def plot_heavy_light_pair(time, 
                          xic, 
                          manual_bd=None, 
                          pred_bd=None,
                          selected_transition=None):
    
    not_selected_transition = np.setdiff1d(np.arange(xic.shape[1]),selected_transition)

    fig, axs = plt.subplots(2, sharex=True)
    for i in range(xic.shape[0]):

        if selected_transition is None:
            axs[i].plot(time, ((-1)**(i+2))*xic[i, :, :].T)
        else:
            axs[i].plot(time, ((-1)**(i+2))*xic[i, selected_transition, :].T, linestyle='solid')
            axs[i].plot(time, ((-1)**(i+2))*xic[i, not_selected_transition, :].T, linestyle='dotted')

        axs[i].set_yticks([])
        # axs[i].set_xticks([])
        # axs[i].axis("off")
        for bd, c, ls in zip([manual_bd, pred_bd], ['black', 'red'], ['dotted', 'dashed']):
            if bd is not None:
                axs[i].axvline(x=bd[0], linestyle=ls, color=c)
                axs[i].axvline(x=bd[1], linestyle=ls, color=c)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, axs