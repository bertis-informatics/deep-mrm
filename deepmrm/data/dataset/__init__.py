from mstorch.data.dataset.base import BaseDataset
from mstorch.utils.logger import get_logger

from ...constant import TIME_KEY, XIC_KEY, HEAVY_PEPTIDE_KEY, LIGHT_PEPTIDE_KEY
from .mrm import MRMDataset
from .prm import PRMDataset

logger = get_logger('DeepMRM')


class DeepMrmDataset(BaseDataset):

    def __init__(self, 
                 metadata_df,
                 pdac_chrom_df=None,
                 scl_chrom_df=None,
                 transform=None):
        super().__init__(metadata_df, transform=transform)
        self.pdac_chrom_df = pdac_chrom_df
        self.scl_chrom_df = scl_chrom_df

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        label_idx = sample[self.metadata_index_name]
        heavy_xic, light_xic = None, None
        if sample['replicate_id'] == 0: 
            # SCL-MC1 dataset
            light_xic = [self.scl_chrom_df.loc[label_idx, 'light_xic']]
            heavy_xic = [self.scl_chrom_df.loc[label_idx, 'heavy_xic']]
        else:
            # PDAC-SIT dataset
            c_df = self.pdac_chrom_df.loc[label_idx, :]
            m = c_df['is_heavy']
            heavy_xic = c_df.loc[m, 'XIC'].tolist()
            light_xic = c_df.loc[~m, 'XIC'].tolist()
        
        sample[XIC_KEY] = {
                LIGHT_PEPTIDE_KEY: light_xic, 
                HEAVY_PEPTIDE_KEY: heavy_xic }

        if self.transform:
            sample = self.transform(sample)

        return sample

