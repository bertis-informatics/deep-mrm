import numpy as np

from deepmrm.constant import XIC_KEY
import torch.utils.data
#
# model_input = {
#     'XIC': [
#         {
#             'light': [ 
#                 ([0,1,2,3] , [64, 23, 35, 12]),
#             ]
#             'heavy': [ 
#                 ([0,1,2,3] , [64, 23, 35, 12]),
#             ]
#         }
#     ]
# }

# model_output = {
# }

class DeepMrmInputDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data = list()
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        sample = {
            self.metadata_index_name: idx,
            XIC_KEY: self.data[idx]
        }
        if self.transform:
            sample = self.transform(sample)            

        return sample

    def add_xic_data(self, xic_data):
        assert isinstance(xic_data, XicData)
        self.data.append(xic_data.data)

    def __len__(self):
        return len(self.data)

    @property
    def metadata_index_name(self):
        return 'index'


class XicData(object):

    def __init__(self):
        self.data = {
            'light': list(),
            'heavy': list()
        }

    def add_xic_pair(self, 
            light_xic_time, light_xic_intensity, 
            heavy_xic_time, heavy_xic_intensity):

        self.data['light'].append(
            (light_xic_time, light_xic_intensity)
        )
        self.data['heavy'].append(
            (heavy_xic_time, heavy_xic_intensity)
        )
