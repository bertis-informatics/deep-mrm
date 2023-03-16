from mstorch.data.dataset.base import BaseDataset
from ...constant import XIC_KEY


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

