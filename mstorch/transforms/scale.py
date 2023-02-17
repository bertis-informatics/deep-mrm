import torch


class ChannelMinMaxScaler(torch.nn.Module):
    """Transforms each channel to the range [0, 1].

    Args:
        torch ([type]): [description]
    """

    def __init__(self, target_key='XIC'):
        super().__init__()
        self.target_key = target_key
    
    def forward(self, sample):

        tensor = sample[self.target_key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not tensor.is_floating_point():
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

        # if self.separate_channel:
        #     # scaling per channel
        #     channel_max = tensor.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        #     channel_min = tensor.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        # else:
        #     # global scaling
        #     channel_max, channel_min = tensor.max(), tensor.min()
        channel_max, channel_min = tensor.max(), tensor.min()

        denom = channel_max - channel_min
        # to avoid divide-by-zero
        denom[denom == 0.0] = 1.0 
        scale = 1.0 / denom
        tensor.sub_(channel_min).mul_(scale)

        sample[self.target_key] = tensor

        return sample
