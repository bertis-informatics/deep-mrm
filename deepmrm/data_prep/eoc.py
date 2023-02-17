from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from deepmrm import data_dir, get_yaml_config
from deepmrm.constant import RT_KEY
from deepmrm.utils.skyline_parser import parse_skyline_file
from pyopenms import AASequence


_conf = get_yaml_config()
conf = _conf['EOC']
root_dir = Path(conf['ROOT_DIR'])

MZML_DIR = root_dir / conf['MZML_DIR']
SKY_DIR = root_dir / conf['SKYLINE_DIR']


def get_transition_df():
    transition_file = root_dir / conf['TRANSITION_FILE']
    trans_df = pd.read_csv(transition_file).dropna(how='all', axis=1)
    trans_df['modified_sequence'] = trans_df['modified_sequence'].apply(
                                        lambda s : AASequence.fromString(s).toString())
    return trans_df


def load_skyline_data():

    sky_files = sorted(list(SKY_DIR.rglob('*.sky')))
    
    sample_dfs, peptide_dfs, trans_dfs = [], [], []
    for sky_path in sky_files:
        skyline_name = sky_path.stem
        sample_df, peptide_df, trans_df = parse_skyline_file(sky_path)

        sample_df['skyline_name'] = skyline_name
        peptide_df['skyline_name'] = skyline_name
        trans_df['skyline_name'] = skyline_name
        
        sample_dfs.append(sample_df)
        peptide_dfs.append(peptide_df)
        trans_dfs.append(trans_df)

    sample_df = pd.concat(sample_dfs, ignore_index=True)
    peptide_df = pd.concat(peptide_dfs, ignore_index=True)
    trans_df = pd.concat(trans_dfs, ignore_index=True)

    peptide_df['modified_sequence'] = peptide_df['modified_sequence'].apply(
                                        lambda s : AASequence.fromString(s).toString())

    trans_df['modified_sequence'] = trans_df['modified_sequence'].apply(
                                        lambda s : AASequence.fromString(s).toString())

    return sample_df, peptide_df, trans_df


def remove_wrong_labels(replicate_df, peak_df):
    # [NOTE] patient=YG7119 is related to wrong mass-spec files in Ovarian_cancer_dataset1B.sky
    T = replicate_df.drop_duplicates(
            ['sample_id', 'raw_file', 'sample_name'], keep='first').reset_index(drop=True)
    m = (T['patient_id'] == 'YG7119') & (T['skyline_name'] == 'Ovarian_cancer_dataset1B')
    replicate_df = T[~m].reset_index(drop=True)

    T = peak_df
    m = T['sample_id'].str.startswith('YG7119') & (T['skyline_name'] == 'Ovarian_cancer_dataset1B')
    peak_df = T[~m].reset_index(drop=True)

    return replicate_df, peak_df


def get_metadata_df():
    # trans_df = get_transition_df()
    replicate_df, peak_df, trans_df = load_skyline_data()

    # m = trans_df['modified_sequence'] == 'GVNFDVSK'
    # trans_df[m]

    # mzML file name is not matched with those in skyline
    replace_mzml = {}
    for mzml_name in replicate_df['mzml_file'].unique():
        mzml_path = MZML_DIR / mzml_name
        if not mzml_path.exists():
            mzml_prefix = mzml_path.stem.split('-')[0]
            alt_mzml = list(MZML_DIR.rglob(f'**/{mzml_prefix}*.mzML'))
            replace_mzml[mzml_name] = alt_mzml[0].name
    replicate_df['mzml_file'] = replicate_df['mzml_file'].replace(replace_mzml)

    cols = ['protein_name', 'sequence', 'is_heavy', 'product_ion', 'product_charge']
    trans_df = trans_df.drop_duplicates(cols, keep='first').sort_values(cols).reset_index(drop=True)
    trans_df = trans_df.drop(columns=['skyline_name'])

    cols = ['skyline_name', 'sample_id', 'mzml_file']
    meta_df = peak_df.merge(replicate_df[cols], on=['skyline_name', 'sample_id'], how='inner')

    # [NOTE] patient YG7119 is linked to wrong mass-spec files in skyline files
    # m = meta_df['sample_id'].str.startswith('YG7119')
    # meta_df = meta_df[~m]
    for col in ['start_time', RT_KEY, 'end_time']:
        meta_df[col] = meta_df[col].astype(np.float32)*60

    meta_df = meta_df.reset_index(drop=True)
    
    return meta_df, trans_df


# meta_df, trans_df = get_metadata_df()