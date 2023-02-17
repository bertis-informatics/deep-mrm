from pathlib import Path
import pandas as pd
import numpy as np

from pyopenms import MSExperiment


class MRMExperiment(MSExperiment):

    def __init__(self) -> None:
        super().__init__()
        self._exp_meta_df = None
        self._transition_df = None
        
        self.precursor_ion_charges = [2, 3]
        self.product_ion_charges = [1]
        
        self.min_product_length = 1
        self.max_product_length = 99
        self.min_win = 0.2 # minutes 
        self.min_transition_size = 3 # number of transitions per precursor ion


    def get_match_df(self, sequences, retention_times=None, mz_tolerance=0.08):
        
        if retention_times is None:
            retention_times = [None]*len(sequences)

        all_matches = []
        for idx, (seq, rt) in enumerate(zip(sequences, retention_times)):
            # if idx == 147:
            #     break
            
            match_df = self._find_matches(
                                sequence=seq, 
                                retention_time=rt, 
                                mz_tolerance=mz_tolerance)
            match_df['compound_idx'] = idx
            all_matches.append(match_df)

        all_match_df = pd.concat(all_matches, ignore_index=True)
        all_match_df['product_ion_charge'] = all_match_df['product_ion_charge'].astype(np.int64)
        all_match_df['cleavage_index'] = all_match_df['cleavage_index'].astype(np.int64)

        # Merge with chromatogram's metadata
        all_match_df = all_match_df.merge(self.metadata_df, left_on='chrom_idx', right_index=True)

        return all_match_df

    @property
    def metadata_df(self):
        if self._exp_meta_df is None:
            self._exp_meta_df = pd.DataFrame([[
                    chrom.getPrecursor().getMZ(), 
                    chrom.getProduct().getMZ(),
                    chrom[0].getRT()/60,
                    chrom[chrom.size()-1].getRT()/60
                ] for chrom in self.getChromatograms()
            ], columns=['precursor_mz', 'product_mz', 'min_rt', 'max_rt'])
        
        return self._exp_meta_df

    @property
    def min_mz(self):
        exp_meta_df = self.metadata_df
        return min(exp_meta_df['precursor_mz'].min(), exp_meta_df['product_mz'].min())

    @property
    def max_mz(self):
        exp_meta_df = self.metadata_df
        return max(exp_meta_df['precursor_mz'].max(), exp_meta_df['product_mz'].max())

    
    def find_chromatogram_index(self, q1, q3, mz_tolerance=1e-1):
        meta_df = self.metadata_df

        mz_diff = (meta_df['precursor_mz'] - q1).abs() + (meta_df['product_mz'] - q3).abs()
        chrom_idx = np.argmin(mz_diff)

        if mz_diff[chrom_idx] > mz_tolerance:
            raise ValueError(f'Cannot find chromatogram for ({q1}, {q3}) transition')

        return int(chrom_idx)


    def _find_matches(self, sequence, retention_time=None, mz_tolerance=0.05):

        exp_meta_df = self.metadata_df
        max_mz = self.max_mz + 1
        max_len = min(sequence.size(), self.max_product_length)
        min_len = max(self.min_product_length, 1)        
        min_win = self.min_win

        product_ion_df = pd.DataFrame([
                [i, product_charge, sequence.getSuffix(i).getMZ(product_charge)] 
                    for i in range(min_len, max_len) 
                        for product_charge in self.product_ion_charges
            ], columns=['cleavage_index', 'charge', 'mz'])
        product_ion_df = product_ion_df.sort_values('mz').reset_index(drop=True)
        product_ion_df = product_ion_df[product_ion_df['mz'] < max_mz]

        all_matches = []
        for precursor_charge in self.precursor_ion_charges:
            # for each precursor charge, 
            # find matches between acquired-chroms and product-ions
            precursor_mz = sequence.getMZ(precursor_charge)
            m1 = (exp_meta_df['precursor_mz'] - precursor_mz).abs() < mz_tolerance

            if retention_time:
                m2 = (exp_meta_df['min_rt'] - min_win < retention_time) & \
                       (retention_time < exp_meta_df['max_rt'] + min_win )
                m3 = m1 & m2
                if np.sum(m3) >= self.min_transition_size:
                    m1 = m3

            selected_exp_df = exp_meta_df[m1]
            
            matches = []
            for chrom_idx, chrom_row in selected_exp_df.iterrows():
                q3 = chrom_row['product_mz']
                j = product_ion_df['mz'].searchsorted(q3)
                if j < product_ion_df.shape[0]:
                    mz_error = np.abs(product_ion_df.loc[j, 'mz'] - q3)
                    if mz_error < mz_tolerance:
                        match = [chrom_idx, mz_error, precursor_charge] + product_ion_df.loc[j, ['charge', 'cleavage_index']].tolist()
                        matches.append(match)
                if j-1 >= 0:
                    mz_error = np.abs(product_ion_df.loc[j-1, 'mz'] - q3)
                    if mz_error < mz_tolerance:
                        match = [chrom_idx, mz_error, precursor_charge] + product_ion_df.loc[j-1, ['charge', 'cleavage_index']].tolist()
                        matches.append(match)
            matched_transitions = len(matches)
            if matched_transitions > 0 and matched_transitions < self.min_transition_size:
                print(f'{sequence} ({precursor_charge}+) has only {matched_transitions} chromatograms being matched. Ignored.')
            else:
                all_matches.extend(matches)

        cols = [
            'chrom_idx', 'mz_error', 
            'precursor_ion_charge', 
            'product_ion_charge', 
            'cleavage_index'
        ]
        match_df = pd.DataFrame.from_records(all_matches, columns=cols)
        # remove duplicated chromatograms 
        match_df = match_df.sort_values(['precursor_ion_charge', 'cleavage_index', 'mz_error'])\
                           .drop_duplicates(['precursor_ion_charge', 'cleavage_index'], keep='first')

        return match_df

                