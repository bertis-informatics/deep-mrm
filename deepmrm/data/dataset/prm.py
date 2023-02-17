from re import S
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from mstorch.data.mass_spec import MassSpecDataReaderFactory
from deepmrm.data.dataset.mrm import MRMDataset
from deepmrm.constant import TIME_KEY, XIC_KEY, HEAVY_PEPTIDE_KEY, LIGHT_PEPTIDE_KEY


class PRMDataset(MRMDataset):

    def __init__(self,
                 file_path,
                 transition_data,
                 metadata_df=None,
                 rt_window_size=None,
                 transform=None):

        super().__init__(file_path, transition_data, metadata_df=metadata_df, transform=transform)
        self.xic = None
        self.rt_window_size = rt_window_size

    def load_data(self, save_path):
        self.xic_data = joblib.load(save_path)

    def extract_data(self, tolerance):
        # [TODO] determine the width of the extraction window based on the resolving power of mass analyzer
        # width = 2 * mz * sqrt(mz) / sqrt(referenceMz) / resolution
        # https://skyline.ms/announcements/home/support/thread.view?rowId=22685
        # https://skyline.ms/announcements/home/support/thread.view?entityId=a6ebc95f-05ec-1033-bad2-da202582f4ca&_anchor=22458#row:22458

        ms_reader = self.ms_reader
        rt_window_size = self.rt_window_size

        use_ref_rt = ((self.transition_data.rt_col is not None) and 
                        (rt_window_size is not None))
        xics = dict()
        for spectrum in tqdm(ms_reader.read_spectra(), total=ms_reader.num_spectra):
            if spectrum.get_ms_level() != 2:
                continue
            iso_win = spectrum.get_isolation_window()
            rt = spectrum.get_retention_time()

            min_rt, max_rt = None, None
            if use_ref_rt:
                min_rt = max(0, rt - rt_window_size*0.5)
                max_rt = rt + rt_window_size*0.5
            
            trans_df = self.transition_data.find_peptides(
                                                iso_win.min_mz, iso_win.max_mz, 
                                                min_rt, max_rt)

            for idx, row in trans_df.iterrows():
                prod_mz = row[self.transition_data.product_mz_col]
                prod_peak = spectrum.find_peak_with_tolerance(prod_mz, tolerance)
                intensity = prod_peak.intensity if prod_peak else 0
                if idx not in xics:
                    xics[idx] = list()
                xics[idx].append((rt, intensity))

        # transpose from [Nx2] to [2xN]
        xics = {k: np.array(v).T for k, v in xics.items()}

        # associate transition XICs with peptide
        xics_new = {}
        for peptide_id, heavy_df, light_df in self.transition_data.iterate_peptide():
            xics_new[peptide_id] = {
                HEAVY_PEPTIDE_KEY: [xics[i] for i in heavy_df.index],
                LIGHT_PEPTIDE_KEY: [xics[i] for i in light_df.index]
            }

        self.xic_data = xics_new

    def save_data(self, save_path):
        joblib.dump(self.xic_data, save_path)

    def __getitem__(self, idx):
        #sample = super().__getitem__(idx)
        sample = super(MRMDataset, self).__getitem__(idx)

        peptide_id = sample[self.peptide_id_col]
        xics = self.xic_data[peptide_id]
        sample[XIC_KEY] = xics
        if self.transform:
            sample = self.transform(sample)
        return sample



