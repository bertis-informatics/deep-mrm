import math
import torch.nn
from torchvision import transforms as T
import numpy as np


class BatchXics(torch.nn.Module):

    def __init__(self, min_transitions=3, size_divisible=32):
        super().__init__()
        self.min_transitions = min_transitions
        self.size_divisible = size_divisible

    def normalize(self, xic_array):
        input = torch.from_numpy(xic_array)
        # normalize each transition pair
        denom = input.max(dim=-1, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        xic_tensor = input / denom.clamp_min(1e-12).expand_as(input)

        # # heatmap normalization
        # separate_channel = True
        # if separate_channel:
        #     # scaling per channel
        #     channel_max = xic_tensor.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
        #     channel_min = xic_tensor.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0]
        # else:
        #     # global scaling
        #     channel_max, channel_min = xic_tensor.max(), xic_tensor.min()

        # denom = channel_max - channel_min
        # # to avoid divide-by-zero
        # denom[denom == 0.0] = 1.0 
        # scale = 1.0 / denom
        # xic_tensor.sub_(channel_min).mul_(scale)
        
        return xic_tensor

    def forward(self, device, xic_list, targets=None):
        
        batch_shapes = np.array([xic.shape for xic in xic_list])
        batch_shape = np.max(batch_shapes, axis=0)
        
        # the length of XIC matrix should be divisable by stride such that
        # the grid of feature map is consistently mapped to each point in XIC
        stride = float(self.size_divisible)
        batch_shape[-1] = int(math.ceil(float(batch_shape[-1]) / stride) * stride)

        # The height of XIC matrix should be >= specified min_transitions 
        # which should be >= 2d kernel height in CNN layer
        batch_shape[1] = max(self.min_transitions, batch_shape[1])
        
        # final batch shape
        batch_shape = [len(xic_list)] + batch_shape.tolist()

        batched_xics = torch.zeros(batch_shape, dtype=torch.float32, device=device)
        for i, xic_array in enumerate(xic_list):
            xic_tensor = self.normalize(xic_array)
            batched_xics[i, :, :xic_tensor.shape[1], :xic_tensor.shape[2]].copy_(xic_tensor)
            # [NOTE] make sure there are at least 3 XICs
            if xic_tensor.shape[1] < self.min_transitions:
                for j in range(xic_tensor.shape[1], self.min_transitions):
                    batched_xics[i, :, j, :xic_tensor.shape[2]].copy_(xic_tensor[:, 0, :])

        if targets is not None:
            targets = [
                {k: torch.from_numpy(v).to(device) for k, v in target.items()}
                    for target in targets
            ]
            return batched_xics, targets

        return batched_xics

