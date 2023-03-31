from multiprocessing import Pool
import joblib
from pathlib import Path
import re

import pandas as pd
import numpy as np
from deepmrm import get_yaml_config
from deepmrm.constant import RT_KEY
from deepmrm.utils.skyline_parser import parse_skyline_file
from pyopenms import AASequence

from mstorch.utils.logger import get_logger

logger = get_logger('DeepMRM')

_conf = get_yaml_config()
conf = _conf['P100_PRM']
root_dir = Path(conf['ROOT_DIR'])

MZML_DIR = root_dir / conf['MZML_DIR']
SKY_DIR = root_dir / conf['SKYLINE_DIR']
XIC_DIR = root_dir/ 'xic'


def get_metadata_df():
    dataset = 2
    sky_files = sorted(list(SKY_DIR.rglob('*.sky')))
    sky_path = [p for p in sky_files if f'Data Set {dataset}' in p.stem][0]
    sample_df, peptide_df, trans_df = parse_skyline_file(sky_path)

    cols = ['protein_name', 'modified_sequence', 'is_heavy', 'product_charge', 'product_ion']
    T = trans_df.groupby(cols)['protein_name'].count()
    assert np.sum(T > 1) == 0

    # Select heavy and light transition pairs
    cols = ['protein_name', 'modified_sequence', 'product_ion', 'product_charge']
    m = trans_df['is_heavy']
    heavy_trans_df = trans_df.loc[m, :].merge(trans_df.loc[~m, cols], how='inner', on=cols)
    light_trans_df = trans_df.loc[~m, :].merge(trans_df.loc[m, cols], how='inner', on=cols)
    trans_df = pd.concat((light_trans_df, heavy_trans_df), ignore_index=True)
    trans_df = trans_df.sort_values(['protein_name', 'modified_sequence', 'is_heavy', 'product_ion']).reset_index(drop=True)

    # trans_df['modified_sequence'] = trans_df['modified_sequence'].replace({'.0': ''}, regex=True)    
    # mod_labels = []
    # for s in trans_df['modified_sequence'].unique():
    #     mod_labels.extend(re.findall(r'\[.*?\]', str(s)))
    # mod_labels = set([s[2:-1] for s in mod_labels])
    # rep_dic = {m: str(int(np.round(float(m)))) for m in mod_labels}
    # trans_df['modified_sequence'] = trans_df['modified_sequence'].replace(rep_dic, regex=True)    
    trans_df['modified_sequence'] = trans_df['modified_sequence'].apply(
                                        lambda s : AASequence.fromString(s).toString())
    
    peptide_df['modified_sequence'] = peptide_df['modified_sequence'].apply(
                                            lambda s : AASequence.fromString(s).toString())
    

    cols = ['sample_id', 'mzml_file']
    meta_df = peptide_df.merge(sample_df[cols], on=['sample_id'], how='inner')
    
    # change time-unit from minutes to seconds
    for col in ['start_time', RT_KEY, 'end_time']:
        meta_df[col] = meta_df[col]*60

    meta_df = meta_df.reset_index(drop=True)    

    return meta_df, trans_df


# def extract_xic(mzml_path):

#     meta_df, all_trans_df = get_metadata_df()
#     peptide_id_col = 'modified_sequence'
#     mz_tolerance = 0.5

#     # mzml_path = MZML_DIR / 'F20131112_LINCS_PC3-Rep3-04_daunorubicin_01.mzML'
#     save_path = XIC_DIR / f'{mzml_path.stem}.pkl'
#     if save_path.exists():
#         logger.warning(f'[{mzml_path.stem}] already extracted. Skip')
#         return

#     logger.info(f'[{mzml_path.stem}] start extraction')
#     # meta_df.loc[meta_df['mzml_file'] == mzml_path.name, :]
#     target_peptides = meta_df.loc[meta_df['mzml_file'] == mzml_path.name, peptide_id_col].unique() 
    
#     m = np.in1d(all_trans_df[peptide_id_col], target_peptides)
#     trans_df = all_trans_df[m].sort_values(['precursor_mz', 'is_heavy', 'product_mz'])
    
#     exp = OnDiscMSExperiment()
#     _ = exp.openFile(str(mzml_path))

#     xics = dict()
#     for i in range(exp.getNrSpectra()):
#         # i += 1
#         spec = exp.getSpectrum(i)
#         if spec.getMSLevel() != 2:
#             continue
#         precursor = spec.getPrecursors()[0]
#         precursor_mz = precursor.getMZ()
#         rt = spec.getRT()
        
#         iso_win = precursor_mz + np.array(
#                     [-precursor.getIsolationWindowLowerOffset(), precursor.getIsolationWindowUpperOffset()])

#         m = (trans_df['precursor_mz'] > iso_win[0]) & (trans_df['precursor_mz'] < iso_win[1])
#         for idx, row in trans_df[m].iterrows():
#             index = spec.findHighestInWindow(row['product_mz'], mz_tolerance, mz_tolerance)
#             intensity = spec[index].getIntensity() if index >= 0 else 0
#             if idx not in xics:
#                 xics[idx] = list()
#             xics[idx].append((rt, intensity))
        
#         if i % 3000 == 0:
#             logger.info(f'[{mzml_path.stem}] {i+1} spectra processed')

#     # transpose from [Nx2] to [2xN]
#     xics = {k: np.array(v).T for k, v in xics.items()}
#     xics_new = {}
#     for seq, sub_df in trans_df.groupby('modified_sequence'):
#         m = sub_df['is_heavy']
#         xics_new[seq] = {
#             'heavy': [xics[i] for i in sub_df.index[m]],
#             'light': [xics[i] for i in sub_df.index[~m]]
#         }
    
#     logger.info(f'[{mzml_path.stem}] complete extraction')
#     joblib.dump(xics_new, save_path)
    
#     # xic_data = joblib.load(save_path)
#     return


# def run_batch_xic_extraction(n_jobs=8):
#     mzml_files = sorted(list(MZML_DIR.rglob('*.mzML')))

#     with Pool(n_jobs) as p:
#         p.map(extract_xic, mzml_files)

# if __name__ == "__main__":
#     run_batch_xic_extraction(n_jobs=8)




from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data.transition import TransitionData
from deepmrm.data.dataset.prm import PRMDataset
from multiprocessing import Pool

meta_df, all_trans_df = None, None
transition_data = None

def extract_xic(mzml_path):
    global meta_df, all_trans_df
    global transition_data

    tolerance = Tolerance(20, ToleranceUnit.PPM)
    # mzml_path = MZML_DIR / mzml_fname
    print(mzml_path)
    save_path = XIC_DIR / f'{mzml_path.stem}.pkl'
    if save_path.exists():
        return

    ds = PRMDataset(mzml_path, transition_data)
    ds.extract_data(tolerance)
    ds.save_data(save_path)
    print(save_path)
    return

def run_batch_xic_extraction(n_jobs=8):
    global meta_df, all_trans_df
    global transition_data
    meta_df, all_trans_df = get_metadata_df()
    transition_data = TransitionData(all_trans_df, peptide_id_col='modified_sequence')
    mzml_files = sorted(list(MZML_DIR.rglob('*.mzML')))
    with Pool(n_jobs) as p:
        p.map(extract_xic, mzml_files)
        
if __name__ == "__main__":
    run_batch_xic_extraction(n_jobs=8)



# mzml_path = '/mnt/c/Users/jungk/Documents/Dataset/DeepMRM_sample_data/PRM/F20131203_LINCS_HL60-Rep1-01_anisomycin_01.mzML'    

# tolerance = Tolerance(20, ToleranceUnit.PPM)
#     # mzml_path = MZML_DIR / mzml_fname
# print(mzml_path)
# from mstorch.data.mass_spec.reader.mzml_reader import MzMLFileReader

# reader = MzMLFileReader(mzml_path, in_memory=False)
# for spectrum in reader.read_spectra():
#     if spectrum.get_ms_level() != 2:
#         continue
#     break

# ds = PRMDataset(mzml_path, transition_data)
# ds.extract_data(tolerance)


# from deepmrm.data.dataset.mrm import MRMDataset

# issubclass(MRMDataset, ds)

# type(ds) == MRMDataset