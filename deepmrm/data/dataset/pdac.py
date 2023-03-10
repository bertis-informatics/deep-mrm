import numpy as np

from mstorch.data.dataset.base import BaseDataset
from ...constant import TIME_KEY, XIC_KEY, HEAVY_PEPTIDE_KEY, LIGHT_PEPTIDE_KEY


class DeepMrmDataset(BaseDataset):

    def __init__(self, 
                 metadata_df,
                 pdac_xic,
                 scl_xic,
                 transform=None):
        super().__init__(metadata_df, transform=transform)
        self.pdac_xic = pdac_xic
        self.scl_xic = scl_xic

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        if sample['replicate_id'] == 0: 
            key = (sample['patient_id'], sample['peptide_id'])
            # SCL-MC1 dataset
            xic = self.scl_xic[key]
        else:
            # PDAC-SIT dataset
            key = (int(sample['patient_id']), int(sample['replicate_id']), sample['peptide_id'])
            xic = self.pdac_xic[key]
        
        sample[XIC_KEY] = xic

        if self.transform:
            sample = self.transform(sample)

        return sample



# class PeakQualityDataset(BaseDataset):

#     def __init__(self, 
#                  metadata_df,
#                  pdac_xic,
#                  transform=None):
#         super().__init__(metadata_df, transform=transform)
#         self.pdac_xic = pdac_xic


#     def __getitem__(self, idx):
#         sample = super().__getitem__(idx)
        
#         # PDAC-SIT dataset
#         key = (int(sample['patient_id']), int(sample['replicate_id']), sample['peptide_id'])
#         xic = self.pdac_xic[key]
        
#         selected_trans_idx = np.random.permutation(range(3))[:2]
#         label = np.sum([sample[f'manual_frag_quality_t{k}'] for k in selected_trans_idx]) == 2

#         # label = sample['manual_quality']
#         # l1 = sample['manual_frag_quality_t1']
#         # l2 = sample['manual_frag_quality_t2']
#         # l3 = sample['manual_frag_quality_t3']
        
#         # s = l1+l2+l3
#         # if s == 2:
#         #     ## one of transitions is poor
#         #      = [k for k in range(3) if sample[f'manual_frag_quality_t{k}']]
#         # elif s== 1:
#         #     raise ValueError('incorrect labels')
#         # else:
#         #     pass

#         sample[XIC_KEY] = xic
#         sample['manual_quality'] = label

#         if self.transform:
#             sample = self.transform(sample)

#         return sample