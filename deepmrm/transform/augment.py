import torch
import numpy as np
from scipy.interpolate import CubicSpline

from torchvision import transforms as T
from torchvision.transforms import functional as F

from ..constant import XIC_KEY, RT_KEY, TIME_KEY
from .transition import TransitionRankShuffle


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given chromatogram randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample
        
        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        rt = sample[RT_KEY]
        xic = sample[XIC_KEY]

        new_xic = np.flip(xic, axis=2).copy()
        new_start_time = time_points[0] + (time_points[-1] - end_time)
        new_rt = time_points[0] + (time_points[-1] - rt)
        new_end_time = time_points[0] + (time_points[-1] - start_time)

        sample['start_time'] = new_start_time
        sample['end_time'] = new_end_time
        sample[XIC_KEY] = new_xic
        sample[RT_KEY] = new_rt

        return sample




class RandomResizedCrop(torch.nn.Module):
    """Crop a random portion of XIC and resize it with a random scale
       RT will be used when cropping
    """
    
    def __init__(self, p=0.5, 
                    min_scale=0.5, 
                    max_scale=5,
                    min_elution_period=3,
                    max_elution_period=180,
                    max_size=900,
                    cycle_time=0.5,
                    manual_bd_only=True):
    
        super().__init__()
        self.p = p
        self.max_size = max_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ep_min = min_elution_period
        self.ep_max = max_elution_period
        self.cycle_time = cycle_time
        self.manual_bd_only = manual_bd_only

    def forward(self, sample):

        if self.manual_bd_only and sample['manual_boundary'] == 0:
            return sample

        if torch.rand(1) > self.p:
            return sample

        try:
            time_points = sample[TIME_KEY]
            start_time = sample['start_time']
            end_time = sample['end_time']
            rt = sample[RT_KEY]
            xic = sample[XIC_KEY]
            xic_len = xic.shape[-1]
            ep = end_time - start_time

            peak_points_idx = np.interp([start_time, rt, end_time], time_points, np.arange(xic_len)) 
            min_ep = max(ep*self.min_scale, self.ep_min)
            max_ep = min(ep*self.max_scale, self.ep_max)
            new_ep = np.random.uniform(min_ep, max_ep, size=1)[0]
            
            scale = new_ep / ep

            max_segment = float(self.max_size)/float(scale)
            peak_segment =  peak_points_idx[-1] - peak_points_idx[0]
            extra_segment = max_segment - peak_segment
            
            crop_st_idx = np.random.uniform(peak_points_idx[0]-extra_segment, peak_points_idx[0]-5)
            crop_st_idx = max(crop_st_idx, 0)
            crop_ed_idx = np.random.uniform(peak_points_idx[-1]+5, crop_st_idx+max_segment)
            crop_ed_idx = min(crop_ed_idx, xic_len)
            crop_st_idx, crop_ed_idx = int(crop_st_idx), int(crop_ed_idx)

            new_time = time_points[crop_st_idx:crop_ed_idx]
            new_len = int(len(new_time)*scale)
            new_xic = xic[:, :, crop_st_idx:crop_ed_idx]
            new_peak_points_idx =  peak_points_idx - crop_st_idx

            resized_xic = F.resize(torch.from_numpy(new_xic), 
                            (new_xic.shape[1], new_len), 
                            interpolation=T.InterpolationMode.BILINEAR)

            resized_time = np.linspace(new_time[0], new_time[0]+new_len*self.cycle_time, new_len)
            resized_peak_points_idx = new_peak_points_idx*scale
            resized_peak_times = np.interp(resized_peak_points_idx, np.arange(new_len), resized_time)
        except:
            print(f'start_time: {start_time}, end_time: {end_time}, scale: {scale}')
            return sample
        
        sample['start_time'] = resized_peak_times[0]
        sample[RT_KEY] = resized_peak_times[1]
        sample['end_time'] = resized_peak_times[2]
        
        # new_time = np.linspace(time_points[0], time_points[-1], new_len)
        sample[XIC_KEY] = resized_xic.numpy()
        sample[TIME_KEY] = resized_time

        return sample


class TimeWarping(torch.nn.Module):

    def __init__(self, sigma=0.2, knot=4, p=0.5):
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def forward(self, sample):
        
        if sample['manual_quality'] == 0:
            return sample

        if torch.rand(1) > self.p:
            return sample
        
        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        xic = sample[XIC_KEY]
        rt = sample[RT_KEY]
        xic_len = xic.shape[-1]

        orig_steps = np.arange(xic_len)
        step_size = (xic_len-1) / (self.knot+1)

        random_warps = np.random.normal(loc=1, scale=self.sigma, size=self.knot+2)
        warp_steps = [0] + [step_size] * (self.knot+1)

        knot_orig = np.cumsum(warp_steps) # np.linspace(0, xic_len-1., num=self.knot+2)
        knot_new = np.cumsum(warp_steps * random_warps)

        warped_steps = CubicSpline(knot_orig, knot_new, bc_type='clamped')(orig_steps)
        
        scale = (xic_len-1)/warped_steps[-1]
        time_warped = scale*warped_steps

        new_boundary_idx = np.interp([start_time, end_time, rt], time_points, time_warped)
        new_times = np.interp(new_boundary_idx, orig_steps, time_points)
        
        if ((new_boundary_idx[1] - new_boundary_idx[0] < 6) or 
                (new_times[1] - new_times[0] < 3)):
            return sample

        # knot_new = scale*knot_new
        xic_warped = np.zeros_like(xic)
        for i in range(xic.shape[0]):
            for j in range(xic.shape[1]):
                xic_warped[i, j, :] = np.interp(orig_steps, time_warped, xic[i, j, :], left=0, right=0)

        sample['start_time'] = new_times[0]
        sample['end_time'] = new_times[1]
        sample[RT_KEY] = new_times[2]
        sample[XIC_KEY] = xic_warped

        return sample



class RandomErase(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample
        
        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        rt = sample[RT_KEY]
        xic = sample[XIC_KEY]
        
        idx = np.searchsorted(time_points, start_time) - 1
        if idx > 10:
            c = np.random.randint(xic.shape[0])
            t = np.random.randint(xic.shape[1])
            st, ed = np.sort(np.random.randint(0, idx, size=2))
            xic[c, t, st:ed] = 0
        
        idx = np.searchsorted(time_points, end_time) + 1
        if idx < len(time_points) - 10:
            c = np.random.randint(xic.shape[0])
            t = np.random.randint(xic.shape[1])
            st, ed = np.sort(np.random.randint(idx, len(time_points), size=2))
            xic[c, t, st:ed] = 0
        
        sample[XIC_KEY] = xic

        return sample


class RandomRTShift(torch.nn.Module):

    def __init__(self, 
                 shift_scale_lb=0.5,
                 shift_scale_ub=4.0,
                 p=0.5):
        super().__init__()
        self.p = p
        self._lower = shift_scale_lb 
        self._upper = shift_scale_ub

    def forward(self, sample):
        
        # if sample['manual_quality'] == 0:
        #     return sample

        if torch.rand(1) > self.p:
            return sample

        time_points = sample[TIME_KEY]
        start_time = sample['start_time']
        end_time = sample['end_time']
        xic = sample[XIC_KEY]
        manual_peak_quality = sample['manual_peak_quality']

        ep = end_time - start_time 
        shift_time = np.random.uniform(ep*self._lower, ep*self._upper)
        shift_index = int( shift_time * (len(time_points)/(time_points[-1] - time_points[0])) )
        
        # select a channel randomly
        c = np.random.choice([0, 1], p=[0.7, 0.3])

        if torch.rand(1) > 0.5:
            xic[c, :, :-shift_index] = xic[c, :, shift_index:]
            xic[c, :, -shift_index:] = 0
        else:
            xic[c, :, shift_index:] = xic[c, :, :-shift_index]
            xic[c, :, :shift_index] = 0

        sample[XIC_KEY] = xic
        sample['manual_quality'] = 0
        sample['manual_boundary'] = 0
        sample['manual_peak_quality'] = np.zeros(manual_peak_quality.shape)

        return sample        



class TransitionJitter(torch.nn.Module):

    def __init__(self, 
                 additive_noise_scale=0, 
                 multiplicative_noise_scale=1e-4,
                 p=0.5):
        super().__init__()
        self.additive_noise_scale = additive_noise_scale
        self.multiplicative_noise_scale = multiplicative_noise_scale
        self.p = p
        self.additive_noise = self.additive_noise_scale > 0

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample

        xic_array = sample[XIC_KEY]
        # number of heavy & light pairs
        num_transition_pairs = xic_array.shape[1]
        # additive_noise = torch.rand(1) > 0.5
        random_noise_scale = np.random.rand()

        if self.additive_noise:
            scale = random_noise_scale * self.additive_noise_scale
            # print(f'additive_noise - {scale}')
        else:
            scale = random_noise_scale * self.multiplicative_noise_scale                    
            # print(f'multiplicative_noise - {scale}')

        # Apply transition-specific noise
        for i in range(2):
            for j in range(num_transition_pairs):
                xic = xic_array[i, j, :]
                xic_center = np.median(xic)
                quantile_range = np.quantile(xic, [0.25, 0.75])
                xic_scale = quantile_range[1] - quantile_range[0]

                if self.additive_noise:
                    noise_data = (scale) * np.random.normal(
                                                loc=xic_center,
                                                scale=xic_scale,
                                                size=(xic.shape[-1]))
                else:
                    noise_data = (scale * xic) * np.random.normal(
                                                loc=0, 
                                                scale=xic_scale,
                                                size=(xic.shape[-1]))
                new_xic = xic + noise_data
                xic_array[i, j, :] = new_xic.clip(0)

        ## Apply additive noise
        # noise_data = self.additive_noise_scale * np.random.normal(
        #                                 loc=np.mean(xic),
        #                                 scale=np.std(xic),
        #                                 size=(num_transition_pairs, xic.shape[-1]))
        # sample[XIC_KEY] = xic + np.tile(noise_data, (2, 1)).reshape(xic.shape)
        sample[XIC_KEY] = xic_array

        return sample        


class MultiplicativeJitter(torch.nn.Module):

    def __init__(self, 
                 noise_scale_ub=1e-3,
                 noise_scale_lb=1e-5,
                 p=0.5):
        super().__init__()
        self.p = p
        # self._upper = np.log(noise_scale_ub)
        # self._lower = np.log(noise_scale_lb)
        upper = np.log(noise_scale_ub)
        lower = np.log(noise_scale_lb)
        self._loc = (upper+lower)*0.5
        self._scale = (upper-lower)/4


    def get_random_scale(self):
        # rd = np.random.rand()
        # rd = rd *(self._upper - self._lower) + self._lower
        rd = np.random.normal(self._loc, self._scale)

        return np.exp(rd)

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample

        xic_array = sample[XIC_KEY]
        # number of heavy & light pairs
        num_transition_pairs = xic_array.shape[1]
        scale = self.get_random_scale()
        
        # Apply transition-specific noise
        for i in range(2):
            for j in range(num_transition_pairs):
                xic = xic_array[i, j, :]
                # xic_center = np.median(xic)
                quantile_range = np.quantile(xic, [0.25, 0.75])
                xic_scale = quantile_range[1] - quantile_range[0]
                noise_data = (scale * xic) * np.random.normal(
                                            loc=0, 
                                            scale=xic_scale,
                                            size=(xic.shape[-1]))
                new_xic = xic + noise_data
                xic_array[i, j, :] = new_xic.clip(0)

        sample[XIC_KEY] = xic_array
        return sample

class ShuffleSignals(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        # if sample['manual_quality'] == 0:
        #     return sample

        if torch.rand(1) > self.p:
            return sample

        xic = sample[XIC_KEY]
        manual_peak_quality = sample['manual_peak_quality']

        # target_channel = np.random.choice([0, 1], p=[0.7, 0.3])
        target_channel = np.random.choice([0, 1])

        # shuffle each XIC separately
        for t in range(xic.shape[1]):
            np.random.shuffle(xic[target_channel, t, :])

        sample[XIC_KEY] = xic
        sample['manual_quality'] = 0
        sample['manual_boundary'] = 0
        sample['manual_peak_quality'] = np.zeros(manual_peak_quality.shape)

        return sample
    


class ReplicateTransitionWithNoise(torch.nn.Module):

    def __init__(self, 
                 multiplicative_noise_scale=1e-4,
                 p=0.5):
        super().__init__()
        self.multiplicative_noise_scale = multiplicative_noise_scale
        self.p = p

    def forward(self, sample):
        
        if sample['manual_quality'] != 1:
            return sample
        
        if torch.rand(1) > self.p:
            return sample
        
        xic_array = sample[XIC_KEY]
        manual_peak_quality = sample['manual_peak_quality']
        
        indexes = np.where(manual_peak_quality > 0)[0]
        summed_xic = xic_array[:, indexes, :].sum(axis=1, keepdims=True)
        scale = self.multiplicative_noise_scale * (np.random.rand() + 0.5)

        # Apply transition-specific noise
        for i in range(2):
            xic = summed_xic[i, 0, :]
            quantile_range = np.quantile(xic, [0.25, 0.75])
            xic_scale = quantile_range[1] - quantile_range[0]
            noise_data = (scale * xic) * np.random.normal(
                                                loc=0, 
                                                scale=xic_scale,
                                                size=(xic_array.shape[-2:]))
            new_xic = xic + noise_data
            xic_array[i, :, :] = new_xic


        sample[XIC_KEY] = xic_array

        return sample           