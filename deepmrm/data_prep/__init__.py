import pandas as pd
import numpy as np 

from deepmrm.data_prep import pdac, scl
from deepmrm.constant import RT_KEY

def get_metadata_df(use_scl=False):
    
    label_df, pdac_chrom_df = pdac.get_metadata_df()

    if use_scl:
        scl_df, scl_chrom_df = scl.get_metadata_df()

        scl_df.index = pd.RangeIndex(100000, 100000+scl_df.shape[0])
        scl_chrom_df.index = scl_df.index
        # duplicate
        for i in range(3):
            scl_df[f'manual_frag_quality_t{i+1}'] = True
            scl_df[f'manual_frag_ratio_t{i+1}'] = scl_df['manual_ratio']
            scl_df[f'manual_light_frag_auc_t{i+1}'] = scl_df['light_auc']
            scl_df[f'manual_heavy_frag_auc_t{i+1}'] = scl_df['heavy_auc']

        label_df = pd.concat((label_df, scl_df))
    else:
        scl_chrom_df = None

    for i in range(3):
        label_df[f'manual_frag_quality_t{i+1}'] = label_df[f'manual_frag_quality_t{i+1}'].astype(int)

    # time unit from minutes to seconds
    time_columns = ['light_rt', 'heavy_rt', 'start_time', 'end_time']
    for col in time_columns:
        label_df[col] = label_df[col]*60
    label_df[RT_KEY] = (label_df['heavy_rt'] + label_df['light_rt'])*0.5

    return label_df, pdac_chrom_df, scl_chrom_df