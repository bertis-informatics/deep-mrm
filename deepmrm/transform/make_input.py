import itertools
import math

import torch.nn
from torchvision import transforms as T
import numpy as np
import pandas as pd

from deepmrm.constant import (
    TARGET_KEY, XIC_KEY, RT_KEY, TIME_KEY,
    HEAVY_PEPTIDE_KEY, LIGHT_PEPTIDE_KEY,
    TRAIN_LC_WINDOW
)



class MakeInput(torch.nn.Module):
    
    def __init__(self, 
                 force_resampling=True, 
                 use_rt=False, 
                 ref_rt_key=RT_KEY, 
                 window_size=TRAIN_LC_WINDOW,
                 cycle_time=0.5):

        super().__init__()
        self.window_size = window_size
        self.force_resampling = force_resampling
        self.use_rt = use_rt
        self.ref_rt_key = ref_rt_key
        self.cycle_time = cycle_time

    def _get_sample_points(self, sample):

        light_xics= sample[XIC_KEY][LIGHT_PEPTIDE_KEY]
        heavy_xics = sample[XIC_KEY][HEAVY_PEPTIDE_KEY]
        time_window = np.array([(xic[0][0], xic[0][-1]) for xic in itertools.chain(light_xics, heavy_xics)])
        min_rt = time_window[:, 0].max()
        max_rt = time_window[:, 1].min()
        if max_rt < min_rt:
            # isolation windows were not correct in instruments
            min_rt = time_window[:, 0].min()
            max_rt = time_window[:, 1].max()

        if ((self.ref_rt_key not in sample) or pd.isna(sample[self.ref_rt_key]) or (not self.use_rt)):
            n_points = int((max_rt - min_rt)/self.cycle_time)
            return np.linspace(min_rt, max_rt, n_points, dtype=np.float32)
        
        rt = sample[self.ref_rt_key] # RT in [minutes]
        if rt < self.window_size*0.5:
            return np.linspace(0, self.window_size, int(self.window_size/self.cycle_time), dtype=np.float32)
        else:
            min_rt = max(min_rt, rt - self.window_size*0.5)
            max_rt = min(max_rt, min_rt + self.window_size)
            return np.linspace(min_rt, max_rt, int((max_rt-min_rt)/self.cycle_time), dtype=np.float32)

    def forward(self, sample):

        raw_xic = sample[XIC_KEY]
        light_xics= raw_xic[LIGHT_PEPTIDE_KEY]
        heavy_xics = raw_xic[HEAVY_PEPTIDE_KEY]
        num_light_trans = len(light_xics) 
        num_heavy_trans = len(heavy_xics)
        if num_light_trans != num_heavy_trans: 
            raise ValueError(f'The pairs of heavy and light XICs do not match. {sample}') 

        len_xics = [len(xic[0]) for xic in itertools.chain(light_xics, heavy_xics)]  
        equal_xic_len = len(set(len_xics)) == 1

        if self.force_resampling or not equal_xic_len:
            # XIC lengths are different
            time_points = self._get_sample_points(sample)
            xic_heatmap = np.zeros((2, num_light_trans, len(time_points)), dtype=np.float32)
            for j, k in enumerate([LIGHT_PEPTIDE_KEY, HEAVY_PEPTIDE_KEY]):
                for i in range(num_light_trans):
                    x, y = raw_xic[k][i]
                    xic_heatmap[j, i, :] = np.interp(time_points, x, y, left=0, right=0)
        elif equal_xic_len:
            time_points = light_xics[0][0]
            # (transitions, length)
            light_heatmap = np.stack([xic[1] for xic in raw_xic[LIGHT_PEPTIDE_KEY]])
            heavy_heatmap = np.stack([xic[1] for xic in raw_xic[HEAVY_PEPTIDE_KEY]])
            xic_heatmap = np.stack((light_heatmap, heavy_heatmap)).astype(np.float32)
        else:
            raise ValueError(f'XICs have different lengths: {len_xics}')

        sample[XIC_KEY] = xic_heatmap
        sample[TIME_KEY] = time_points
        
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # NOTE: returning torch.tensor objects will casue errors like
        #   RuntimeError: unable to open shared memory object </torch_7065_1741927742_3060>
        #   in read-write mode: Too many open files (24)
        
        return sample


def inside_time_window(time_points, time):
    return (time_points[0]-1 < time) and (time < time_points[-1]+1)


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


class TransitionDuplicate(torch.nn.Module):

    def __init__(self, max_num_duplication=8):
        super().__init__()
        self.max_num_duplication = max_num_duplication

    def forward(self, sample):

        xic = sample[XIC_KEY]
        num_transition_pairs = xic.shape[1]
        if num_transition_pairs < 4:
            return sample

        num_duplication = min(math.comb(num_transition_pairs, 3), self.max_num_duplication)
        index = np.arange(num_transition_pairs)
        # [TODO] boundary transitions between duplicates should not be overlapped
        xic_duplicated = [xic]
        prev_index = index
        for _ in range(num_duplication):
            bd_index = prev_index[-2:]
            bd_index2 = np.random.choice(np.setdiff1d(index, bd_index), replace=False, size=2)
            cur_index = np.concatenate((
                bd_index2, np.random.permutation(np.setdiff1d(index, bd_index2))
            ))
            xic_duplicated.append(xic[:, cur_index, :])
            prev_index = cur_index
        # xic_duplicated = [
        #     xic[:, np.random.permutation(index), :]
        #         for _ in range(num_duplication)
        # ]
        new_xic = np.concatenate(xic_duplicated, axis=1)
        sample[XIC_KEY] = new_xic

        return sample        


class SelectSegment(torch.nn.Module):

    def __init__(self, max_num_duplication=8):
        super().__init__()
        self.max_num_duplication = max_num_duplication

    def forward(self, sample):

        #xics = self.xic_data[peptide_id]
        xics = sample[XIC_KEY]

        # [TODO] select one from multiple XIC segments without label
        new_xics = dict()
        for key, xic_set in xics.items():
            # key = 'light'
            # xic_set = xics[key]
            xic_set = xics[key]
            ref_time = xic_set[0][0]

            time_diff = ref_time[1:] - ref_time[:-1]
            split_points = list(np.where(time_diff > 5)[0])
            split_points = [-1] + split_points + [len(ref_time)]

            start_time_idx = np.searchsorted(ref_time, sample['start_time'])
            ii = np.searchsorted(split_points, start_time_idx) - 1

            st_idx = split_points[ii] + 1
            ed_idx = split_points[ii+1]
            new_xics[key] = [xic[:, st_idx:ed_idx] for xic in xic_set]
            
        sample[XIC_KEY] = new_xics

        return sample