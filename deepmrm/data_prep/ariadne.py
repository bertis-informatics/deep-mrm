from pathlib import Path
import pandas as pd
import numpy as np
from deepmrm import data_dir, get_yaml_config
from mstorch.utils.logger import get_logger

from deepmrm.utils.skyline_parser import parse_skyline_file

logger = get_logger('DeepMRM')

_conf = get_yaml_config()
conf = _conf['Ariadne']
root_dir = Path(conf['ROOT_DIR'])


def get_trans_df():

    df = pd.read_excel(root_dir / 'supplementary/other/transList.xlsx')
    df = df.rename(columns={
                'Q1': 'precursor_mz', 
                'Q3': 'product_mz', 
                'transition_group_id': 'peptide_id'})
    df['is_heavy'] = df['isotype'] == 'heavy'
    df['peptide_id'] = df.apply(lambda x: f'{x["stripped_sequence"]}_{x["prec_z"]}+', axis=1)

    m = df['protein_name'].str.startswith('RT-Kit')
    df = df[~m].reset_index(drop=True)

    return df


def get_mzml_files():
    mzml_dir = root_dir / 'mzML'
    
    dfs = [
        get_skyline_result_df('Noisy'),
        get_skyline_result_df('SmoothBack'),
        get_skyline_result_df('Smooth')
    ]

    mzml_files = [
        mzml_dir / f'{x}.mzML' for df in dfs for x in df['File Name'].unique()
    ]

    return mzml_files


def get_skyline_result_df(dataset):
    
    assert dataset in ['Noisy', 'SmoothBack', 'Smooth']

    df = pd.read_csv(root_dir / f'Results_Skyline_{dataset}.tsv', sep='\t')
    m = df['Protein Name'].str.startswith('RT-Kit')
    df = df[~m].reset_index(drop=True)

    df['peptide_id'] = df.apply(lambda x: f'{x["Peptide Sequence"]}_{x["Precursor Charge"]}+', axis=1)

    def extract_heavy_amount(replicate_name):
        s = replicate_name.split('_')
        
        if s[0] == 'at':
            fmol = float(s[1])/1000
        elif s[0] == 'ft':
            fmol = float(s[1])
        else:
            raise ValueError()

        return fmol
    
    df['Heavy peptide abundance (fmole)'] = df['Replicate Name'].apply(extract_heavy_amount)
    df['File Name'] = df['File Name'].apply(lambda x : x[:-4])

    # there are duplicated replicate numbers:
    # (e.g. 20121126_AQ1_10ft_1_rep1.RAW,  20121126_AQ1_10ft_2_rep1.RAW)
    df = df.drop_duplicates(['peptide_id', 'File Name', 'Heavy peptide abundance (fmole)'], keep='first')
    df = df.rename(columns={'RatioLightToHeavy': 'RatioLightToHeavy_Skyline'})

    return df








