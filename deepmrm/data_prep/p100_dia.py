from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

from deepmrm import data_dir, get_yaml_config
from pyopenms import AASequence
from mstorch.utils.logger import get_logger

from deepmrm.utils.skyline_parser import parse_skyline_file

logger = get_logger('DeepMRM')

_conf = get_yaml_config()
conf = _conf['P100_DIA']
root_dir = Path(conf['ROOT_DIR'])

MZML_DIR = root_dir / conf['MZML_DIR']
SKY_DIR = root_dir / conf['SKYLINE_DIR']
XIC_DIR = root_dir/ 'xic'
LABEL_TYPES = conf['LABEL_TYPES']
SKY_FILES = {
        'Skyline': 'LINCS_P100_DIA_Plate34_Quant_AllTransitions_Top5SkyPB.sky',
        'Manual': 'LINCS_P100_DIA_Plate34_Quant_OptTrans_ManualPB.sky', 
        'AvG': 'LINCS_P100_DIA_Plate34_Quant_AllTransitions.sky'
    }








def get_spectral_lib_df():
    spec_lib_path = SKY_DIR / 'Manual/SV_P100_DIA_CE30_PurePeptides.blib'
    con = sqlite3.connect(spec_lib_path)

    lib_df = pd.read_sql_query("SELECT peptideModSeq, retentionTime as ref_rt FROM RefSpectra", con)

    lib_df['peptideModSeq'] = lib_df['peptideModSeq'].apply(lambda s : s.replace('S[+121.976896]', '(Acetyl).S(Phospho)'))

    def remove_label_mod(ori_mod_seq_str):
        s = ori_mod_seq_str.replace('K[+8.014199]', 'K').replace('R[+10.008269]', 'R')
        return AASequence.fromString(s).toString()
        
    lib_df['modified_sequence'] = lib_df['peptideModSeq'].apply(remove_label_mod)
    lib_df['modified_sequence_heavy'] = lib_df['peptideModSeq'].apply(lambda s : AASequence.fromString(s).toString())

    lib_df['ref_rt'] *= 60

    # return lib_df[['modified_sequence', 'modified_sequence_heavy', 'ref_rt']]
    return lib_df



def normalize_seq_str(seq_ser):
    mod_map = {
        'A[+42]': '.(Acetyl)A',
        'S[+121.976896]': '.(Acetyl)S(Phospho)',
        'S[+122]': '.(Acetyl)S(Phospho)',
    }
    def fix_mod(s):
        for k, v in mod_map.items():
            s = s.replace(k, v)
        return s

    seq_ser = seq_ser.apply(fix_mod)
    seq_ser = seq_ser.apply(lambda s : AASequence.fromString(s).toString())
    return seq_ser


def get_transition_df():

    lib_df = get_spectral_lib_df()
    trans_df = pd.read_csv(SKY_DIR / 'AvG' / 'Peptide Transition List.csv')
    trans_df['Peptide Modified Sequence'] = normalize_seq_str(trans_df['Peptide Modified Sequence'])
    trans_df['is_heavy'] = trans_df['Isotope Label Type'] == 'heavy'

    col_rename = {
        'Protein Name': 'protein_name',
        'Peptide Modified Sequence': 'modified_sequence',
        'Precursor Mz': 'precursor_mz',
        'Precursor Charge': 'precursor_charge', 
        'Product Mz': 'product_mz', 
        'Product Charge': 'product_charge', 
        'Fragment Ion': 'product_ion', 
        'Fragment Ion Type': 'product_ion_type',  
        'Fragment Ion Ordinal': 'cleavage_index', 
        'Loss Neutral Mass': 'neutral_loss',
        'Losses': 'neutral_loss_desc',
        'Library Rank': 'spec_lib_rank', 
        'Library Intensity': 'spec_lib_intensity',
        'is_heavy': 'is_heavy'
    }
    trans_df = trans_df.drop(
                    columns=trans_df.columns.difference(list(col_rename))
                ).rename(columns=col_rename)

    # Pairing heavy and light transitions
    cols = ['protein_name', 'modified_sequence', 'product_ion', 'product_charge']
    m = trans_df['is_heavy']
    heavy_trans_df = trans_df.loc[m, :].merge(trans_df.loc[~m, cols], how='inner', on=cols)
    light_trans_df = trans_df.loc[~m, :].merge(trans_df.loc[m, cols], how='inner', on=cols)
    trans_df = pd.concat((light_trans_df, heavy_trans_df), ignore_index=True)
    trans_df = trans_df.sort_values([
                'protein_name', 'modified_sequence', 
                'is_heavy', 'product_ion_type', 'cleavage_index', 
                'product_charge', 'neutral_loss']).reset_index(drop=True)
    trans_df = trans_df.merge(
                    lib_df[['modified_sequence', 'modified_sequence_heavy', 'ref_rt']], 
                    on='modified_sequence', 
                    how='left')

    def get_sequence_obj(row):
        seq_str = row['modified_sequence_heavy'] if row['is_heavy'] else row['modified_sequence']
        return AASequence.fromString(seq_str)

    def calc_precursor_mz(row):
        seq_str = row['modified_sequence_heavy'] if row['is_heavy'] else row['modified_sequence']
        precursor_mz = AASequence.fromString(seq_str).getMZ(row['precursor_charge'])
        return precursor_mz

    # check precursor m/z from sequence string    
    T = trans_df.apply(calc_precursor_mz, axis=1)
    assert np.all((trans_df['precursor_mz'] - T).abs() < 1e-3)

    trans_df['seq_openms'] = trans_df.apply(get_sequence_obj, axis=1)
    trans_df = trans_df.drop(columns=['modified_sequence_heavy'])
    #trans_df.shape
    return trans_df
    
    # sample_df, trans_df = get_transition_df()
    # sequence_obj = trans_df.iloc[0, -1]
    # frag_df = TransitionData.generate_transitions(
    #                         sequence_obj, 
    #                         precursor_ion_charges=[2],
    #                         product_ion_charges = [1, 2],
    #                         min_product_length=4,
    #                         max_product_length=99,
    #                         min_mz=150,
    #                         max_mz=1500)

def get_sample_df():
    label_type = 'Skyline'
    sky_path = SKY_DIR /label_type/ SKY_FILES[label_type]
    sample_df, _, _ = parse_skyline_file(sky_path)
    sample_df = sample_df[['replicate_name', 'mzml_file']]
    sample_df.columns = ['sample_id', 'mzml_file']
    return sample_df

def get_quant_df():
    
    dfs = []
    for label_type in LABEL_TYPES:
        #label_type = 'Manual'
        sky_path = SKY_DIR /label_type/ SKY_FILES[label_type]
        _, peptide_df, _ = parse_skyline_file(sky_path)
        peptide_df['modified_sequence'] = normalize_seq_str(peptide_df['modified_sequence'])
        for col in ['start_time', 'end_time', 'RT']:
            peptide_df[col] = peptide_df[col]*60                

        m = peptide_df['is_heavy'] 
        h_pep = peptide_df.loc[m, ['sample_id', 'modified_sequence', 'dotp']]
        l_pep = peptide_df.loc[~m, ['sample_id', 'modified_sequence', 'dotp']]

        peptide_df = h_pep.merge(l_pep, on=['sample_id', 'modified_sequence'], suffixes=('_heavy', '_light'))
        peptide_df['sample_id'] = peptide_df['sample_id'].str[:-3]
        peptide_df = peptide_df.set_index(['modified_sequence', 'sample_id'])
        cols = [(label_type, col) for col in peptide_df.columns]
        peptide_df.columns = pd.MultiIndex.from_tuples(cols)

        ratio_df = pd.read_csv(SKY_DIR / label_type / 'Peptide Ratio Results2.csv')
        ratio_df['Peptide Modified Sequence'] = normalize_seq_str(ratio_df['Peptide Modified Sequence'])
        ratio_df = ratio_df[ratio_df['Isotope Label Type'] == 'heavy'].drop(columns=['Isotope Label Type'])
        # mins to seconds
        for col in ['Peptide Retention Time', 'Min Start Time', 'Max End Time']:
            ratio_df[col] = ratio_df[col]*60

        col_dic = {
                # 'Protein': 'protein',
                'Replicate': 'sample_id',
                # 'Peptide': 'sequence',
                'Peptide Modified Sequence': 'modified_sequence',
                'Ratio To Standard': (label_type, 'ratio'),
                'Transition Count': (label_type, 'transition_count'),
                'Peptide Retention Time': (label_type, 'RT'),
                # 'is_heavy': (label_type, 'is_heavy'),
                'Min Start Time': (label_type, 'start_time'),
                'Max End Time': (label_type, 'end_time'),
                'Total Area': (label_type, 'total_area'),}
        ratio_df = ratio_df.rename(columns=col_dic).set_index(
                            ['modified_sequence', 'sample_id'])
        cols = [k for k in col_dic.values() if type(k) == tuple]
        ratio_df = ratio_df[cols]
        ratio_df.columns = pd.MultiIndex.from_tuples(cols)
        ratio_df = ratio_df.join(peptide_df)
        dfs.append(ratio_df)
        # m = (T[('Manual', 'dotp_heavy')] > 0.5) & (T[('Manual', 'dotp_light')] > 0.7)
        # T[m].shape[0] / T.shape[0]

    df = pd.concat(dfs, axis=1)
        
    return df

def load_figure3_data():
    base_df = pd.read_excel(SKY_DIR / 'fig3_data.xlsx', sheet_name=0)
    base_df['modified_sequence'] = normalize_seq_str(base_df['PeptideModifiedSequence'])
    non_filter_df = pd.read_excel(SKY_DIR / 'fig3_data.xlsx', sheet_name=1)
    non_filter_df['modified_sequence'] = normalize_seq_str(non_filter_df['PeptideModifiedSequence'])

    key_cols = [
        'modified_sequence', 'IsotopeLabelType', 'PrecursorCharge', 'FileName'
    ]
    base_cols = key_cols + ['Manual_validation', 'unoptimized_Skyline'] 
    cols = key_cols + ['Avantgarde']
    df = base_df[base_cols].merge(non_filter_df[cols], on=key_cols, how='left')

    filter_df = pd.read_excel(SKY_DIR / 'fig3_data.xlsx', sheet_name=2)
    filter_df['modified_sequence'] = normalize_seq_str(filter_df['PeptideModifiedSequence'])
    filter_df['is_filtered'] = True
    cols = key_cols + ['is_filtered']

    df = df.merge(filter_df[cols], on=key_cols, how='left')
    df['is_filtered'] = df['is_filtered'].fillna(False)
    df['sample_id'] = df['FileName'].replace({'_P0034_': '_P-0034_'}, regex=True)
    df = df.drop(columns=['IsotopeLabelType', 'FileName'])

    return df


# RT 체크
def _check_rt():
    q_df = get_quant_df()
    t_df = get_transition_df()
    t_df = t_df.drop_duplicates('modified_sequence')

    tmp = q_df.loc[:, ('Manual', 'RT')].reset_index(drop=False)
    tmp.columns = ['modified_sequence', 'sample_id', 'manual_rt']

    tmp_sky = q_df.loc[:, ('Skyline', 'RT')].reset_index(drop=False)
    tmp_sky.columns = ['modified_sequence', 'sample_id', 'sky_rt']

    tmp = tmp.merge(tmp_sky, on=['modified_sequence', 'sample_id'])
    tmp = tmp.merge(t_df[['modified_sequence', 'ref_rt']], on='modified_sequence')

    tmp['sky_manual_diff'] = (tmp['manual_rt'] - tmp['sky_rt']).abs()
    tmp['ref_manual_diff'] = (tmp['manual_rt'] - tmp['ref_rt']).abs()
    tmp['sky_ref_diff'] = (tmp['sky_rt'] - tmp['ref_rt']).abs()

    # tmp.loc[tmp['rt_diff'] > 60*4.8, 'rt_diff'].max()
    # m = tmp['sky_manual_diff'] > 10 
    # tmp[m]
    # tmp.loc[m, 'modified_sequence'].unique()
    return tmp


from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data.transition import TransitionData
from deepmrm.data.dataset.prm import PRMDataset
from multiprocessing import Pool
from deepmrm.data_prep import p100_dia

all_trans_df = get_transition_df()
transition_data = TransitionData(all_trans_df, rt_col='ref_rt')

def extract_xic(mzml_fname):
    rt_window_size = 20*60
    tolerance = Tolerance(20, ToleranceUnit.PPM)
    mzml_path = p100_dia.MZML_DIR / mzml_fname
    print(mzml_path)
    save_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
    if save_path.exists():
        return

    ds = PRMDataset(mzml_path, transition_data, rt_window_size=rt_window_size)
    ds.extract_data(tolerance)
    ds.save_data(save_path)
    print(save_path)
    return


def run_batch_xic_extraction(n_jobs=8):
    mzml_files = sorted(list(MZML_DIR.rglob('*.mzML')))
    with Pool(n_jobs) as p:
        p.map(extract_xic, mzml_files)
    
        
# if __name__ == "__main__":
#     run_batch_xic_extraction(n_jobs=8)
    


