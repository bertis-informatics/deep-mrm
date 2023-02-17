import joblib
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from pyopenms import (
    AASequence, Residue, 
    OnDiscMSExperiment, MzMLFile
) 
from deepmrm.data.mrm_experiment import MRMExperiment
from deepmrm import data_dir, get_yaml_config


SILAC_LABELS = {
    'K': '(Label:13C(6)15N(2))',
    'R': '(Label:13C(6)15N(4))',
}

_conf = get_yaml_config()
conf = _conf['OVARIAN']

root_path = Path(conf['ROOT_DIR'])
mzml_paths = [Path(x) for x in conf['MZML_DIR']]
transition_path = root_path / conf['TRANSITION_FILE']

def strip_df(df):
    df.columns = [col.strip() for col in df.columns]
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df


def get_transition_df(exp_id):
    trans_df = strip_df(pd.read_excel(transition_path, sheet_name='all_peptides'))
    trans_df['Protein'] = trans_df['CompoundName'].str[:-3]
    trans_df['Is_Heavy'] = trans_df['CompoundName'].str[-2] == 'H'
    seq_df = strip_df(pd.read_excel(transition_path, sheet_name=f'{exp_id}_seq'))
    trans_df = trans_df.merge(seq_df, how='right', on='Protein') # .drop(columns=['CompoundName', 'Formula', 'AdductPositive'])
    sequences = []
    cterm_label = '(Label:13C(6)15N(4))'
    for _, row in trans_df.iterrows():
        seq_str = row['Peptide'] 
        cterm_label = SILAC_LABELS[seq_str[-1]] if row['Is_Heavy'] else ''
        seq = AASequence.fromString(seq_str + cterm_label)
        precursor_charge = row['z']
        assert np.abs(row['m/z'] - seq.getMZ(precursor_charge)) < 1e-3
        sequences.append(seq)

    trans_df['Sequence'] = sequences
    cols = ['Protein', 'Peptide', 'Sequence','Is_Heavy', 
            'm/z', 'z', 'RT Time (min)', 'Window (min)']

    trans_df = trans_df.sort_values('m/z').reset_index(drop=True)
    return trans_df[cols]
    
exp_id = 2
mzml_dir = mzml_paths[exp_id-1]
trans_df = get_transition_df(exp_id)
mzml_files = list(mzml_dir.rglob('*.mzML'))

mzml_path = str(mzml_files[0])
exp = MRMExperiment()

exp = OnDiscMSExperiment()
# mzml = MzMLFile()
# mzml.load(mzml_path, exp)
_ = exp.openFile(mzml_path)

# chrom = exp.getChromatogram(0)
# x, y = chrom.get_peaks()
ms1_spectra = []
spectra_map = dict()
for i in range(exp.getNrSpectra()):
    spec = exp.getSpectrum(i)
    if spec.getMSLevel() == 1:
        ms1_spectra.append(spec)
        continue
    rt = spec.getRT()/60
    precursor = precursors = spec.getPrecursors()[0]
    precursor_mz = precursor.getMZ()
    if precursor_mz in spectra_map:
        spectra_map[precursor_mz].append(
            spec
        )
    else:
        spectra_map[precursor_mz] = [spec]

# precursors = np.array(sorted(list(spectra)))
        
for spec in ms1_spectra:
    if spec.getRT()/60 > 29:
        break
# trans_df['m/z'].apply(lambda)
x, y = spec.get_peaks()

from matplotlib import pyplot as plt
pre_mz = 586.8115
plt.figure()
plt.plot(x, y)
plt.xlim([pre_mz-1, pre_mz+1])
plt.savefig('tmp.jpg')

# precursors[0].getIsolationWindowLowerOffset()
# 8      P52566     TLLGDGPVVTDPK                         TLLGDGPVVTDPK     False  656.3614  2           18.7             6
# 19      P18206      SLLDASEEAIKK      SLLDASEEAIKK(Label:13C(6)15N(2))      True  656.3659  2           16.0             6

rts = []
mzs = []
pre_mz = 586.8115
mz_tol = pre_mz*10*1e-6
for mz, spectra in spectra_map.items():
    # m = (precursors > 656.35) & (precursors < 656.37)
    if np.abs(mz - pre_mz) < mz_tol:
        for spec in spectra:
            rts.append( spec.getRT() )
            mzs.append( mz )

rts = np.sort(np.array(rts)/60)
# mzs
# rts

# s = 'SLLDASEEAIKK' (Label:13C(6)15N(2))'
# seq = AASequence.fromString(s)
# seq.getMZ(2)