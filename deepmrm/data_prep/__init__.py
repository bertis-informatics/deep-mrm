import pandas as pd
import numpy as np 

from deepmrm.data_prep import pdac, scl
from deepmrm.constant import RT_KEY

def get_metadata_df(use_scl=False):
    use_scl = True
    
    label_df, pdac_xic = pdac.get_metadata_df()

    if use_scl:
        scl_df, scl_xic = scl.get_metadata_df()
        scl_df['replicate_id'] = 0
        for i in range(3):
            scl_df[f'manual_frag_quality_t{i+1}'] = 1

        # scl_df.index = pd.RangeIndex(100000, 100000+scl_df.shape[0])
        cols = scl_df.columns.intersection(label_df.columns)
        label_df = pd.concat((label_df[cols], scl_df[cols]), ignore_index=True)
    else:
        scl_xic = None

    for i in range(3):
        label_df[f'manual_frag_quality_t{i+1}'] = label_df[f'manual_frag_quality_t{i+1}'].astype(int)

    # time unit from minutes to seconds
    time_columns = ['light_rt', 'heavy_rt', 'start_time', 'end_time']
    for col in time_columns:
        label_df[col] = label_df[col]*60
    label_df[RT_KEY] = (label_df['heavy_rt'] + label_df['light_rt'])*0.5

    return label_df, pdac_xic, scl_xic