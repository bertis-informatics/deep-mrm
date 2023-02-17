import numpy as np
import pandas as pd

from deepmrm.constant import TIME_KEY, XIC_KEY, HEAVY_PEPTIDE_KEY, LIGHT_PEPTIDE_KEY
from mstorch.data.mass_spec import MassSpecDataReaderFactory
from mstorch.data.dataset.base import BaseDataset
from mstorch.utils.logger import get_logger

logger = get_logger('DeepMRM')


class MRMDataset(BaseDataset):

    def __init__(self,
                 file_path, 
                 transition_data,
                 metadata_df=None,
                 transform=None):
        
        if metadata_df is None:
            metadata_df = pd.DataFrame(
                            transition_data.get_peptide_ids(), 
                            columns=[transition_data.peptide_id_col])
        
        super().__init__(metadata_df, transform=transform)                 
        
        self.transition_data = transition_data
        self.file_path = file_path
        self._ms_reader = None
        self.chrom_index_map = None

    @property
    def ms_reader(self):
        if self._ms_reader is None:
            in_memory = type(self) == MRMDataset
            self._ms_reader = MassSpecDataReaderFactory.get_mass_spec_reader(self.file_path, in_memory=in_memory)
        return self._ms_reader

    @property
    def peptide_id_col(self):
        return self.transition_data.peptide_id_col

    @property
    def num_targets(self):
        return self.transition_data.num_targets
    
    def find_chromatogram_index(self, q1, q3, tolerance):

        meta_df = self.chrom_meta_df
        mz_tolerance = tolerance.get_mz_tolerance(max(q1, q3))
        mz_diff = (meta_df['precursor_mz'] - q1).abs() + (meta_df['product_mz'] - q3).abs()
        chrom_idx = np.argmin(mz_diff)

        if mz_diff[chrom_idx] > mz_tolerance:
            raise ValueError(f'Cannot find chromatogram for ({q1}, {q3}) transition')

        return int(chrom_idx)        


    def extract_data(self, tolerance):
        
        self.isolation_wins = [
                (chrom.get_isolation_window(q=1), chrom.get_isolation_window(q=3))
                    for chrom in self.ms_reader.read_chromatograms()]
        
        self.chrom_meta_df = pd.DataFrame(
                                [(wins[0].mz, wins[1].mz) for wins in self.isolation_wins],
                                columns=['precursor_mz', 'product_mz'])

        self.chrom_index_map = dict()
        precursor_mz_col = self.transition_data.precursor_mz_col
        product_mz_col = self.transition_data.product_mz_col

        for pep_id, heavy_df, light_df in self.transition_data.iterate_peptide():
            map_ret = []
            num_transitions =  light_df.shape[0]
            for idx in range(num_transitions):
                light_trans, heavy_trans = light_df.iloc[idx], heavy_df.iloc[idx]
                
                try:
                    light_chrom_idx = self.find_chromatogram_index(
                                        light_trans[precursor_mz_col], 
                                        light_trans[product_mz_col], 
                                        tolerance)
                    heavy_chrom_idx = self.find_chromatogram_index(
                                        heavy_trans[precursor_mz_col], 
                                        heavy_trans[product_mz_col], 
                                        tolerance)             
                    map_ret.append((LIGHT_PEPTIDE_KEY, idx, light_chrom_idx))
                    map_ret.append((HEAVY_PEPTIDE_KEY, idx, heavy_chrom_idx))
                except:
                    pass                       
            
            n = heavy_df.shape[0]
            m = int(len(map_ret)*0.5)
            if m != n:
                logger.warning(f'Can find only {m} XICs out of {n} fragements specified for {pep_id}')

            
            if len(map_ret) > 0:
                self.chrom_index_map[pep_id] = map_ret

        logger.info(f'Found XICs for {len(self.chrom_index_map)} out of {self.num_targets} target peptides')

    def __getitem__(self, idx):
        
        sample = super().__getitem__(idx)
        peptide_id = sample[self.peptide_id_col]

        xics = {LIGHT_PEPTIDE_KEY: list(), HEAVY_PEPTIDE_KEY: list()}
        chrom_map = self.chrom_index_map[peptide_id]

        for key, idx, chrom_idx in chrom_map:
            chrom_peaks = self.ms_reader.read_chromatogram(chrom_idx).get_peaks()
            x = [pk.retention_time for pk in chrom_peaks]
            y = [pk.intensity for pk in chrom_peaks]
            xics[key].append((x, y))

        sample[XIC_KEY] = xics

        if self.transform:
            sample = self.transform(sample)

        return sample


def test_mrm_dataset():

    from deepmrm.data_prep import eoc
    mzml_dir = eoc.MZML_DIR
    meta_df, trans_df = eoc.get_metadata_df()
    mzml_files = meta_df['mzml_file'].unique()
    mzml_path = mzml_dir / mzml_files[0]

    ms_reader = MassSpecDataReaderFactory.get_mass_spec_reader(mzml_path)
