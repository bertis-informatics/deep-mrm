import pandas as pd
import numpy as np

class TransitionData(object):

    def __init__(self,
                 transition_df,
                 peptide_id_col='peptide_id',
                 is_heavy_col='is_heavy',
                 precursor_mz_col='precursor_mz',
                 product_mz_col='product_mz',
                 rt_col=None):

        # [TODO] check if transition_df is ordered correctly
        transition_df = transition_df.dropna(how='all').astype({
                'is_heavy': np.bool_, 
                precursor_mz_col: np.float64,
                product_mz_col: np.float64,
            })

        heavy_mask = transition_df[is_heavy_col] == True
        light_mask = transition_df[is_heavy_col] == False
        heavy_trans_df, light_trans_df = transition_df[heavy_mask], transition_df[light_mask]
        if (light_trans_df.shape[0] != heavy_trans_df.shape[0] or 
            light_trans_df[peptide_id_col].nunique() != heavy_trans_df[peptide_id_col].nunique()):
            raise ValueError('Mismatch in light and hevay transition pairs')
        
        self.peptide_id_col = peptide_id_col
        self.is_heavy_col = is_heavy_col
        self.precursor_mz_col = precursor_mz_col
        self.product_mz_col = product_mz_col
        self.rt_col = rt_col
        self.df = transition_df
        
    @property
    def num_targets(self):
        return self.df[self.peptide_id_col].nunique()

    @property
    def num_transitions(self):
        return self.df[self.df[self.is_heavy_col]].shape[0]

    def get_peptide_ids(self):
        return self.df[self.peptide_id_col].unique().tolist()

    def find_peptides(self, min_mz, max_mz, min_rt=None, max_rt=None):
        m = (min_mz < self.df[self.precursor_mz_col]) & (self.df[self.precursor_mz_col] < max_mz)
        if (min_rt  is not None and max_rt  is not None and self.rt_col is not None):
            m &= (min_rt < self.df[self.rt_col]) & (self.df[self.rt_col] < max_rt)
        
        return self.df[m]

    def iterate_peptide(self):
        for peptide_id, sub_df in self.df.groupby(self.peptide_id_col):
            m = sub_df[self.is_heavy_col]
            yield peptide_id, sub_df[m], sub_df[~m]

    def get_product_ion_mz_range(self):
        min_mz = self.df[self.product_mz_col].min()
        max_mz = self.df[self.product_mz_col].max()
        return (min_mz, max_mz)

    @staticmethod
    def generate_transitions(
                        sequence_obj, 
                        precursor_ion_charges=[2, 3],
                        product_ion_charges = [1],
                        min_product_length=2,
                        max_product_length=99,
                        min_mz=150,
                        max_mz=1500):
        if isinstance(precursor_ion_charges, int):
            precursor_ion_charges = [precursor_ion_charges]

        if isinstance(product_ion_charges, int):
            product_ion_charges = [product_ion_charges]            

        max_len = min(sequence_obj.size(), max_product_length)
        min_len = max(min_product_length, 1)        

        product_ion_df = pd.DataFrame([
                [i, product_charge, sequence_obj.getSuffix(i).getMZ(product_charge)] 
                    for i in range(min_len, max_len) 
                        for product_charge in product_ion_charges
            ], columns=['cleavage_index', 'product_ion_charge', 'product_ion_mz'])

        m = (product_ion_df['product_ion_mz'] > min_mz) & ((product_ion_df['product_ion_mz'] < max_mz))
        product_ion_df = product_ion_df[m]

        dfs = []
        for precursor_charge in precursor_ion_charges:
            precursor_mz = sequence_obj.getMZ(precursor_charge)
            tmp_df = product_ion_df.copy()
            tmp_df['precursor_mz'] = precursor_mz
            tmp_df['precursor_ion_charge'] = precursor_charge
            dfs.append(tmp_df)
        dfs = pd.concat(dfs)

        col_ordered = [
                'precursor_mz', 
                'precursor_ion_charge',
                'product_ion_mz',
                'product_ion_charge',
                'cleavage_index',
            ]
        return dfs[col_ordered]

    def get_sequence_objects(self, modified_sequence_col):
        from pyopenms import AASequence
        
        for tup1, tup2 in zip(self.light_trans_df.groupby(self.peptide_id_col), \
                                    self.heavy_trans_df.groupby(self.peptide_id_col)):
            pep_id = tup1[0]
            light_seq = tup1[1][modified_sequence_col]
            heavy_seq = tup2[1][modified_sequence_col]
            light_seq_obj = AASequence.fromString(light_seq)
            heavy_seq_obj = AASequence.fromString(heavy_seq)
            
            yield pep_id, light_seq_obj, heavy_seq_obj



def test_mrm_transition():
    from deepmrm.data_prep.pdac import get_transition_df

    trans_df = get_transition_df()
    trans_data = TransitionData(
        trans_df,
        peptide_id_col='Compound Name',
        is_heavy_col='Heavy',
        precursor_mz_col='Precursor Ion',
        product_mz_col='Product Ion',
        rt_col='Ret Time (min)',
    )

    return trans_data


def test_prm_transition():
    # from deepmrm.data_prep.p100_dia import get_peptide_df
    # peptide_df = get_peptide_df()
    # for idx, row in peptide_df.iterrows():
    #     seq_obj = row['seq_obj']
    #     pre_charge = row['precursor_charge']
    #     TransitionData.generate_transitions(
    #                     seq_obj, 
    #                     precursor_ion_charges=[1, 2],
    #                     product_ion_charges=[pre_charge],
    #                     min_product_length=2,
    #                     max_product_length=99,
    #                     min_mz=150,
    #                     max_mz=1500)
    pass

    