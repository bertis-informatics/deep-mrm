from pathlib import Path
import pandas as pd
from deepmrm import get_yaml_config, data_dir


_conf = get_yaml_config()
conf = _conf['Ariadne']
MZML_DIR = Path(conf['ROOT_DIR']) / 'mzML'
DATA_DIR = data_dir / 'Ariadne'
MULTI_FACTORS = {
    'Noisy': 10, 
    'SmoothBack': 1,
    'Smooth': 1,
}
DATASETS = list(MULTI_FACTORS)

peptide_id_col='peptide_id'


def get_trans_df():
    # df = pd.read_excel(root_dir / 'supplementary/other/transList.xlsx')
    df = pd.read_excel(DATA_DIR / 'transList.xlsx')
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

    dfs = [
        get_skyline_result_df(ds_name) for ds_name in DATASETS
    ]
    mzml_files = [
        MZML_DIR / f'{x}.mzML' for df in dfs for x in df['File Name'].unique()
    ]

    return mzml_files


def get_skyline_result_df(dataset):
    
    assert dataset in DATASETS

    # df = pd.read_csv(root_dir / f'Results_Skyline_{dataset}.tsv', sep='\t')
    df = pd.read_csv(DATA_DIR / f'Results_Skyline_{dataset}.tsv', sep='\t')

    bd_df = pd.read_csv(DATA_DIR / f'Peak Boundaries_{dataset}.csv')
    bd_df['peptide_id'] = bd_df.apply(lambda x: f'{x["Peptide Modified Sequence"]}_{x["Precursor Charge"]}+', axis=1)
    # heavy & light peptides
    bd_df = bd_df.drop_duplicates(['File Name', 'peptide_id'], keep='first')

    m = df['Protein Name'].str.startswith('RT-Kit')
    df = df[~m].reset_index(drop=True)
    df['peptide_id'] = df.apply(lambda x: f'{x["Peptide Sequence"]}_{x["Precursor Charge"]}+', axis=1)
    
    # merge with peak boundaries
    cols = ['File Name', 'peptide_id', 'Min Start Time', 'Max End Time']
    df = df.merge(bd_df[cols], on=['File Name', 'peptide_id'], how='left')
    assert df['Min Start Time'].isnull().sum() == 0
    assert df['Min End Time'].isnull().sum() == 0

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








