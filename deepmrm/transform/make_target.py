import itertools
import math

import torch.nn
from torchvision import transforms as T
import numpy as np
import pandas as pd

from deepmrm.constant import (
    TARGET_KEY, XIC_KEY, RT_KEY, TIME_KEY
)

def inside_time_window(time_points, time):
    return (time_points[0]-1 < time) and (time < time_points[-1]+1)




class MakePeakQualityTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):

        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        xic = sample[XIC_KEY]

        if (not inside_time_window(time_points, start_time) or 
                not inside_time_window(time_points, end_time)):
            raise ValueError(f'start_time({start_time}) or end_time({end_time}) \
                        is outside of time_points [{time_points[0]}-{time_points[-1]}]')

        peak_boundary = np.array([start_time, end_time])
        boundary_idx = np.interp(peak_boundary, time_points, np.arange(len(time_points)))

        selected_trans_idx = np.random.permutation(range(3))[:2]
        label = 1 if (
            np.sum([sample[f'manual_frag_quality_t{k+1}'] for k in selected_trans_idx]) == 2
            ) else 0

        st_idx, ed_idx = np.round(boundary_idx).astype(int)

        sample[XIC_KEY] = xic[:, selected_trans_idx, st_idx:ed_idx]
        sample['manual_quality'] = label
        
        return sample


class MakeTagets(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, sample):

        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        
        if (not inside_time_window(time_points, start_time) or 
                not inside_time_window(time_points, end_time)):
            raise ValueError(f'start_time({start_time}) or end_time({end_time}) \
                        is outside of time_points [{time_points[0]}-{time_points[-1]}]')

        peak_boundary = np.array([start_time, end_time])
        boundary_idx = np.interp(peak_boundary, time_points, np.arange(len(time_points)))

        ####  Assume that non-quantifiable peak groups are background ####
        if sample['manual_quality'] > 0:
            sample[TARGET_KEY] = {
                'boxes': boundary_idx.reshape(1, -1).astype(np.float32),
                'labels': np.array([1], dtype=np.int64)
            }
        else:
            sample[TARGET_KEY] = {
                'boxes': np.zeros((0, 2), dtype=np.float32),
                'labels': np.zeros((0), dtype=np.int64)
            }

        #####################################################################
        # # background: 0, non-quantifiable: 1, quantifiable: 2
        # label_array = np.array([sample['manual_quality']+1], dtype=np.int64)
        
        # # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # # NOTE: returning torch.tensor objects will casue errors like
        # #   RuntimeError: unable to open shared memory object </torch_7065_1741927742_3060>
        # #   in read-write mode: Too many open files (24)
        # sample[TARGET_KEY] = {
        #     'boxes': boundary_idx.reshape(1, -1).astype(np.float32),
        #     'labels': label_array
        # }
        
        return sample
