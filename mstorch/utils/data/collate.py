import numpy as np
from torch.utils.data._utils.collate import default_collate


def _default_collate(key, batch, exclusion_keys):
    ret = [d[key] for d in batch]
    if key in exclusion_keys:
        return ret
    return default_collate(ret)


class SelectiveCollation:
    def __init__(self, inclusion_keys=None, exclusion_keys=None):
        
        if inclusion_keys and exclusion_keys:
            raise ValueError('Both exclusion or inclusion keys cannot be given')        
        
        self.inclusion_keys = inclusion_keys
        self.exclusion_keys = exclusion_keys if exclusion_keys else list()


    def __call__(self, batch):
        all_keys = list(batch[0])
        if self.inclusion_keys:
            exclusion_keys = np.setdiff1d(all_keys, self.inclusion_keys)
        else:
            exclusion_keys = self.exclusion_keys

        return {
            key: _default_collate(key, batch, exclusion_keys) for key in all_keys
        }        

      


# def object_detection_collate_fn(batch):
#     elem = batch[0]
#     ret = dict()
#     for key in elem:
#         if key == 'targets':
#             ret[key] = [d[key] for d in batch]
#         else:
#             ret[key] = default_collate([d[key] for d in batch])
        
#     return ret