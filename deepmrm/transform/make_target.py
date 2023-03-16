import torch.nn
import numpy as np
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
        num_transitions = xic.shape[1]
        manual_peak_quality = sample['manual_peak_quality']

        if (not inside_time_window(time_points, start_time) or 
                not inside_time_window(time_points, end_time)):
            raise ValueError(f'start_time({start_time}) or end_time({end_time}) \
                        is outside of time_points [{time_points[0]}-{time_points[-1]}]')

        peak_boundary = np.array([start_time, end_time])
        boundary_idx = np.interp(peak_boundary, time_points, np.arange(len(time_points)))

        if sample['manual_quality'] == 1 and np.random.rand() > 0.7:
            # make a pair of identical transition
            indexes = np.where(manual_peak_quality > 0)[0]
            selected_trans_idx = list(np.random.choice(indexes, 1))*2
            label = 1
        else:
            selected_trans_idx = np.random.permutation(range(num_transitions))[:2]
            label = 1 if np.sum(manual_peak_quality[selected_trans_idx]) == 2 else 0

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
        # # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # # NOTE: returning torch.tensor objects will casue errors like
        # #   RuntimeError: unable to open shared memory object </torch_7065_1741927742_3060>
        # #   in read-write mode: Too many open files (24)
        # sample[TARGET_KEY] = {
        #     'boxes': boundary_idx.reshape(1, -1).astype(np.float32),
        #     'labels': label_array
        # }
        
        return sample
