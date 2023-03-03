import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from pyopenms import MzMLFile
from deepmrm.data.mrm_experiment import MRMExperiment
from deepmrm import get_yaml_config, data_dir, private_data_dir
from deepmrm.data.dataset import MRMDataset
from deepmrm.data.transition import TransitionData
from mstorch.data.mass_spec.tolerance import Tolerance

_conf = get_yaml_config()
conf = _conf['SCL-MASTOCHK1']

SCL_DIR = Path(conf['ROOT_DIR'])
SCL_MZML_DIR = Path(conf['MZML_DIR'])



def get_transition_df():
    trans_df = pd.read_csv(private_data_dir/'SCL_transition.csv')
    return trans_df


def get_label_df():
    label_df = pd.read_csv(private_data_dir / 'SCL_label.csv')
    return label_df

def _create_chrom_df():

    save_path = private_data_dir / 'SCL_xic.pkl'
    # raw_label_df = load_raw_label_df()
    label_df = get_label_df()
    trans_df = get_transition_df()
    trans_data = TransitionData(trans_df, rt_col='retention_time')
    tmp_df = label_df.drop_duplicates(['patient_id', 'mzML'], keep='first')

    xic_data = dict()
    for idx, row in tmp_df.iterrows():
        sample_id = row['patient_id']
        mzml_file = row['mzML']
        mzml_path = SCL_MZML_DIR / mzml_file
        
        ds = MRMDataset(
            file_path=mzml_path,
            transition_data=trans_data,
        )
        tolerance = Tolerance(100)
        ds.extract_data(tolerance)

        for i in range(len(ds)):
            sample = ds[i]
            pep_id = sample['peptide_id']
            key = (sample_id, pep_id)
            xic_data[key] = sample['XIC']
    joblib.dump(xic_data, save_path)            


def get_metadata_df():

    xic_data = joblib.load(private_data_dir / 'SCL_xic.pkl')
    label_df = get_label_df()

    return label_df, xic_data 



def _read_scl_data(excel_path):

    wiff_files = list(excel_path.parent.rglob('*.wiff'))
    df_ = pd.read_excel(excel_path, skiprows=None, header=[0,1])
    df_.index = df_.iloc[:, 0].apply(lambda x : int(x.split('_')[-1]))

    tmp = wiff_files[0].stem.split('_')
    mzml_prefix = '{}_{}'.format(tmp[0], tmp[1])
    mzml_files = list(SCL_MZML_DIR.rglob('{}*.mzML'.format(mzml_prefix))) 
    d = {int(f.stem.split('_')[-1]): f.name for f in mzml_files}
    mzml_df = pd.DataFrame.from_dict(d, orient='index')
    mzml_df.columns = ['mzML']
    
    if len(mzml_files) != df_.shape[0]:
        print('#mzml_files is not equal to #excel_records')
    
    df_ = df_.join(mzml_df, how='inner').set_index(df_.columns[0])
    df_.index.name = df_.index.name[1]

    return df_


def load_raw_label_df():
    excel_files = [
        p for p in SCL_DIR.rglob('**/*.xlsx') if not p.name.startswith('~$')
    ]

    df = pd.concat([
                _read_scl_data(fpath) for fpath in excel_files])
    return df
