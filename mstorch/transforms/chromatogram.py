import torch.nn
import numpy as np
from scipy.interpolate import CubicSpline

# Time-series augmentation, 
#   https://arxiv.org/pdf/2004.08780.pdf
#   https://github.com/uchidalab/time_series_augmentation
#   https://halshs.archives-ouvertes.fr/halshs-01357973/document
#
# - jittering (noise addition), 
# - rotation (flipping for univariate; rotation for multivariate), 
# - slicing (cropping), 
# - permutation (rearranging slices),
# - scaling (pattern-wise magnitude change), 
# - magnitude warping (smooth element-wise magnitude change), 
# - time warping (time step deformation), 
# - frequency warping (frequency deformation)

#[TODO] TimeStretch
# https://pytorch.org/tutorials/beginner/audio_feature_augmentation_tutorial.html


class ToTensor(torch.nn.Module):

    def __init__(self, xic_key='XIC', channels=2):
        super().__init__()

        self.xic_key = xic_key
        self.channels = channels


    def forward(self, sample):
        
        xic = sample[self.xic_key]

        if self.channels > 1 and xic.shape[0] % self.channels != 0:
            raise ValueError('#XICs should be divisible by #channels')
        
        if self.channels == xic.shape[0]:
            # 1D timeseries, [channel, timesteps]
            new_xic = torch.from_numpy(xic).to( 
                        dtype=torch.get_default_dtype())
        else:
            # 2d image, [channel, height, width]
            # e.g) each channel for light and heavy transitions
            new_xic = torch.from_numpy(
                        xic.reshape(self.channels, -1, xic.shape[1])).to( 
                            dtype=torch.get_default_dtype()
                        )

        # new_xic = np.moveaxis(new_xic, 0, -1)
        # new_xic = to_tensor(new_xic)
        sample[self.xic_key] = new_xic

        return sample


class Resize(torch.nn.Module):

    def __init__(self,
                 window_size=4,
                 time_steps=310,
                 rt_key='rt',
                 xic_key='XIC',
                 time_key='time'):
        super().__init__()
        self.window_size = window_size
        self.time_steps = time_steps
        self.rt_key = rt_key
        self.xic_key = xic_key
        self.time_key = time_key

    def _get_rt(self, sample):

        min_rt = sample[self.time_key][0][0]
        max_rt = sample[self.time_key][0][-1]
        if (max_rt - min_rt < self.window_size):
            return (min_rt + max_rt)*0.5

        return sample[self.rt_key]

    def forward(self, sample):
        
        rt = self._get_rt(sample)
        half_window_size = self.window_size*0.5

        min_rt = rt - half_window_size
        max_rt = rt + half_window_size
        new_time = np.linspace(min_rt, max_rt, num=self.time_steps)

        new_chroms = []
        for x, chrom in zip(sample[self.time_key], sample[self.xic_key]):
            new_chrom = np.interp(new_time, x, chrom, left=0, right=0)
            new_chroms.append(new_chrom)

        sample[self.xic_key] = np.stack(new_chroms)
        sample[self.time_key] = new_time        
        
        return sample


class RandomResizedCrop(Resize):

    def __init__(self,
                 window_size=4,
                 time_steps=310,
                 rt_key='rt',
                 xic_key='XIC',
                 time_key='time'):
        super().__init__(
                 window_size=window_size,
                 time_steps=time_steps,
                 rt_key=rt_key,
                 xic_key=xic_key,
                 time_key=time_key)            

    def _get_rt(self, sample):

        rt = sample[self.rt_key]
        # min_rt = sample[self.time_key][0][0]
        # max_rt = sample[self.time_key][0][-1]        
        # min_rt = max(min_rt+0.4, rt-self.window_size*0.5)
        # max_rt = min(max_rt-0.4, rt+self.window_size*0.5)
        min_rt = max(rt - 1.0, 0)
        max_rt = rt + 1.0
        rt = torch.FloatTensor(1).uniform_(min_rt, max_rt).item()

        return rt


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given chromatogram randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, xic_key='XIC', p=0.5):
        super().__init__()
        self.p = p
        self.xic_key = xic_key

    def forward(self, sample):
        if torch.rand(1) > self.p:
            return sample

        xic = sample[self.xic_key]
        # np.flip returns view, which causes an error when transforming to tensor
        sample[self.xic_key] = np.flip(xic, axis=1).copy()

        return sample




class HeavyLightSwap(torch.nn.Module):
    pass


class TransitionShuffle(torch.nn.Module):
    """ shuffle the orders of transition pairs
        Doesn't change quality and ratio labels
    """

    def __init__(self, xic_key='XIC', p=0.5):
        super().__init__()
        self.xic_key = xic_key
        self.p = p

    def forward(self, sample):

        if torch.rand(1) > self.p:
            return sample

        xic = sample[self.xic_key]
        # number of pairs of light and heavy peptides
        num_transition_pairs = int(len(xic) / 2)
        idx = np.random.permutation(list(range(num_transition_pairs)))
        idx = np.concatenate((idx, num_transition_pairs+idx))
        xic_shuffled = xic[idx, :]
        sample[self.xic_key] = xic_shuffled

        return sample        




class TimeWarping(torch.nn.Module):

    def __init__(self, xic_key='XIC', sigma=0.2, knot=4, p=0.5):
        super().__init__()
        self.xic_key = xic_key

        self.sigma = sigma
        self.knot = knot
        self.p = p

    def forward(self, sample):
        
        if torch.rand(1) > self.p:
            return sample

        xic = sample[self.xic_key]
        xic_len = xic.shape[1]
        
        time_orig_steps = np.arange(xic_len)
        step_size = (xic_len-1) / (self.knot+1)

        random_warps = np.random.normal(loc=1, scale=self.sigma, size=self.knot+2)
        warp_steps = [0] + [step_size] * (self.knot+1)
        
        time_orig = np.cumsum(warp_steps) # np.linspace(0, xic_len-1., num=self.knot+2)
        time_new = np.cumsum(warp_steps * random_warps)

        time_warp_steps = CubicSpline(time_orig, time_new)(time_orig_steps)
        scale = (xic_len-1)/time_warp_steps[-1]
        time_warped = scale*time_warp_steps
        time_new = scale*time_new

        xic_warped = np.zeros_like(xic)
        for i, pat in enumerate(xic):
            xic_warped[i, :] = np.interp(time_orig_steps, time_warped, pat)

        sample[self.xic_key] = xic_warped
        # wapr_map = {x1: x2 for x1, x2 in zip(time_orig, time_new)}
        return sample


def augmentation_test():
    from automrm.data_prep import pdac
    from matplotlib import pyplot as plt
    import joblib

    time, xic = joblib.load(f'{pdac.PDAC_SIT_TGR_DIR}/134.pkl')

    plt.figure()
    for x, y in zip(time, xic):
        plt.plot(x/60, y)
    plt.savefig('temp/temp.jpg')

    sample = {'time': np.array(time)/60, 'XIC': np.array(xic), 'rt': 29}
    t_sample = TimeWarping()(sample)
    # t = RandomResizedCrop()
    # t_sample = TransitionJitter(scale=0.1)(sample)
    # t_sample = RandomHorizontalFlip(p=1.0)(sample)
    # t_sample = TransitionShuffle()(sample)

    time, xic = t_sample['time'], t_sample['XIC']
    plt.figure()
    for x, y in zip(time, xic):
        plt.plot(x/60, y)
    plt.savefig('temp/temp2.jpg')



