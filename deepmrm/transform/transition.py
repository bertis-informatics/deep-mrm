import torch.nn
from torchvision import transforms as T
import numpy as np
import pandas as pd

from deepmrm.constant import RT_KEY, TIME_KEY, XIC_KEY


class TransitionShuffle(torch.nn.Module):
    """ shuffle the orders of transition pairs
        Doesn't change quality and ratio labels
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p


    def forward(self, sample):

        if sample['replicate_id'] == 0:
            return sample

        if torch.rand(1) > self.p:
            return sample

        xic = sample[XIC_KEY]
        # number of pairs of light and heavy peptides
        num_transition_pairs = xic.shape[1]
        if num_transition_pairs < 2:
            return sample

        idx = np.random.permutation(list(range(num_transition_pairs)))
        xic_shuffled = xic[:, idx, :]
        sample[XIC_KEY] = xic_shuffled

        return sample        


class TransitionRankShuffle(torch.nn.Module):
    """
    Make a transition record non-quantifiable 
    Set quality label to 0
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):

        if torch.rand(1) > self.p:
            return sample

        if sample['replicate_id'] == 0:
            return sample
        if sample['manual_quality'] == 0:
            # skip for non-quantifiable sample
            return sample

        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        rt = sample[RT_KEY]
        num_transition_pairs = xic.shape[1]

        if np.sum(sample['manual_peak_quality']) != num_transition_pairs:
            return sample
        
        rt_index = min(int(len(time)*(rt-time[0])/(time[-1]-time[0])), len(time)-1)
        peak_heights = xic[1, :, rt_index]
        sorted_heavy_trans_index = np.argsort( peak_heights )

        # lowest and highest peaks
        low_idx, high_idx = sorted_heavy_trans_index[0], sorted_heavy_trans_index[-1]
        
        # swap selected two rows
        xic[1, [high_idx, low_idx]] = xic[1, [low_idx, high_idx]]

        # make sure their intensities differ significantly 
        xic[1, low_idx, :] *= np.random.uniform(3, 8)
        xic[1, high_idx, :] *= np.random.uniform(0.1, 0.4)

        # update xic with shuffled order
        sample[XIC_KEY] = xic
        
        # change label from quantifiable to non-quantifiable
        sample['manual_quality'] = 0

        return sample

