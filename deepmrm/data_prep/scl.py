import pandas as pd
import numpy as np
from pathlib import Path

from pyopenms import MzMLFile
from deepmrm.data.mrm_experiment import MRMExperiment
from deepmrm import get_yaml_config, data_dir

_conf = get_yaml_config()
conf = _conf['SCL-MASTOCHK1']

SCL_DIR = Path(conf['ROOT_DIR'])
SCL_MZML_DIR = Path(conf['MZML_DIR'])


def read_scl_data(excel_path):

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
                read_scl_data(fpath) for fpath in excel_files])
    return df


def get_transition_df():

    transition_list = [
        ['APOC1', 'APOC1_L', False, 4.3, 526.748, 776.378],
        ['APOC1', 'APOC1_H', True, 4.3, 530.755, 784.393],
        ['CA1', 'CA1_L', False, 7.2, 485.8, 758.441],
        ['CA1', 'CA1_H', True, 7.2, 489.807, 766.455],
        ['CHL1', 'CHL1_L', False, 4.6, 478.78, 744.4],
        ['CHL1', 'CHL1_H', True, 4.6, 483.784, 754.408]
    ]

    cols = [
        'compound_id', 'peptide_code', 
        'is_heavy', 'retention_time',
        'precursor_mz', 'product_mz'
    ]
    trans_df = pd.DataFrame(transition_list, columns=cols)
    return trans_df


def prepare_dataset():

    raw_label_df = load_raw_label_df()
    trans_df = get_transition_df()
    cols = ['sample_id', 'compound_id', 'peptide_type', 
            'auc', 'height', 'rt', 'xic']

    all_data = []
    for idx, row in raw_label_df.iterrows():
        print(idx)
        mzml_file = row['mzML']
        mzml_path = SCL_MZML_DIR / mzml_file

        exp = MRMExperiment()
        mzml = MzMLFile()
        mzml.load(str(mzml_path), exp)
        
        for _, t_row in trans_df.iterrows():
            pep_code = t_row['peptide_code']
            q1 = t_row['precursor_mz'] 
            q3 = t_row['product_mz'] 
            chrom_idx = exp.find_chromatogram_index(q1, q3)
            x, y = exp.getChromatogram(chrom_idx).get_peaks()
            chrom = np.stack((x, y))

            all_data.append([
                row.name,
                t_row['compound_id'],
                'heavy' if t_row['is_heavy'] else 'light',
                row[(pep_code, 'Area')], 
                row[(pep_code, 'Height')], 
                row[(pep_code, 'Retention Time')],
                chrom
            ])

    a_df = pd.DataFrame(all_data, columns=cols)
    #a_df.set_index(['sample_id', 'compound_id', 'peptide_type']).unstack(level=2)
    a_df['xic'] = a_df['xic'].apply(lambda x : x.astype(np.float32)) # reduce size
    a_df.to_pickle(data_dir/'SCL_label_df.pkl')
    return a_df


def get_metadata_df():
    fpath = data_dir/'SCL_label_df.pkl'
    if fpath.exists():
        scl_df = pd.read_pickle(fpath)
    else:
        scl_df = prepare_dataset()

    scl_df = scl_df.set_index(['sample_id', 'compound_id', 'peptide_type']).unstack(level=2)
    scl_df.columns = [f'{col[1]}_{col[0]}' for col in scl_df.columns]
    scl_df['peptide_code'] = scl_df.index.get_level_values(1)
    scl_df['selected_charge'] = 0
    scl_df['ion_order'] = True
    scl_df['manual_ratio'] = scl_df['light_auc'] / scl_df['heavy_auc']
    scl_df['manual_ratio_desc'] = np.nan
    scl_df['patient_id'] = scl_df.index.get_level_values(0)
    scl_df['replicate_id'] = 0
    scl_df['manual_quality'] = 1 # MC1 data is always quantifiable
    scl_df = scl_df.reset_index(drop=True)
    scl_df.index.name = 'label_idx'

    scl_chrom = scl_df[['light_xic', 'heavy_xic']]
    scl_df = scl_df.drop(columns=['light_xic', 'heavy_xic', 'heavy_height', 'light_height'])

    bd_df = pd.read_csv(data_dir / 'scl_peak_boundary.csv')
    scl_df = scl_df.join(bd_df)
    
    return scl_df, scl_chrom
