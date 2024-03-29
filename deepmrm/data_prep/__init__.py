import pandas as pd
import numpy as np 

from deepmrm.data_prep import pdac, scl
from deepmrm.constant import RT_KEY

def get_metadata_df(use_scl=False, only_peak_boundary=False):
    label_df, pdac_xic = pdac.get_metadata_df()

    if use_scl:
        scl_df, scl_xic = scl.get_metadata_df()
        cols = scl_df.columns.intersection(label_df.columns)
        label_df = pd.concat((label_df[cols], scl_df[cols]), ignore_index=True)
    else:
        scl_xic = None

    # time unit from minutes to seconds
    time_columns = ['light_rt', 'heavy_rt', 'start_time', 'end_time']
    for col in time_columns:
        label_df[col] = label_df[col]*60
    label_df[RT_KEY] = (label_df['heavy_rt'] + label_df['light_rt'])*0.5

    if only_peak_boundary:
        m = label_df['manual_boundary'] == 1
        label_df = label_df[m]

    return label_df, pdac_xic, scl_xic