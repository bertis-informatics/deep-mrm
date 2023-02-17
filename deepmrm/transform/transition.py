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

        if sum([sample[f'manual_frag_quality_t{i+1}'] for i in range(3)]) != 3:
            return sample

        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        # rt = sample[RT_KEY]
        rt = sample['heavy_rt']
        rt_index = min(int(len(time)*(rt-time[0])/(time[-1]-time[0])), len(time)-1)
        num_transition_pairs = xic.shape[1]
        
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


class TransitionSelect(torch.nn.Module):
    """Select a single transition that were used in manual quantification
    Change ratio label while keeping quality label
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):

        if torch.rand(1) > self.p:
            return sample

        if sample['replicate_id'] == 0:
            return sample

        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        rt = sample['heavy_rt']
        
        rt_index = min(int(len(time)*(rt-time[0])/(time[-1]-time[0])), len(time)-1)
        
        qualified_trans_idx = np.array([
                j for j in range(3) 
                    if sample[f'manual_frag_quality_t{j+1}'] == 1 and \
                        not pd.isna(sample[f'manual_heavy_frag_auc_t{j+1}'])
            ])

        if len(qualified_trans_idx) == 0:
            return sample
        
        peak_heights = xic[1, qualified_trans_idx, rt_index]
        selected_trans_idx = qualified_trans_idx[np.argmax(peak_heights)]

        new_xic = xic[:, [selected_trans_idx], :]
        # ignore_trans_idx = np.array([(selected_trans_idx+k)%3 for k in range(1, 3)])
        # new_xic = xic.copy()
        # new_xic[ignore_trans_idx, :] = xic[selected_trans_idx, :]
        # new_xic[num_transition_pairs+ignore_trans_idx, :] = xic[num_transition_pairs+selected_trans_idx, :]

        # update xic with shuffled order
        sample[XIC_KEY] = new_xic
        
        # change label
        sample['manual_quality'] = 1

        return sample


class AddTransitions(torch.nn.Module):
    """Select a single transition that were used in manual quantification
    Change ratio label while keeping quality label
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):

        if torch.rand(1) > self.p:
            return sample

        if sample['replicate_id'] == 0:
            return sample

        if sample['manual_quality'] != 1:
            return sample

        xic = sample[XIC_KEY]
        num_trans = xic.shape[1]
        # qualified_trans_idx = np.array([
        #         j for j in range(3) 
        #             if sample[f'manual_frag_quality_t{j+1}'] == 1 and \
        #                 not pd.isna(sample[f'manual_heavy_frag_auc_t{j+1}'])
        #     ])
        num_adds = np.random.randint(1, 6)
        new_xic = np.zeros((xic.shape[0], num_trans+num_adds, xic.shape[-1]), xic.dtype)
        new_xic[:, :num_trans, :] = xic

        for i in range(num_adds):
            scale = np.random.uniform(low=0.33, high=3)
            # k = np.random.choice(qualified_trans_idx, 1)[0]
            k = np.random.randint(0, num_trans)
            new_xic[:, num_trans+i, :] = xic[:, k, :]*scale

        sample[XIC_KEY] = new_xic

        return sample