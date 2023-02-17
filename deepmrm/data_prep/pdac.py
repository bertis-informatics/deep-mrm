from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from pyopenms import (
    AASequence, Residue, 
    OnDiscMSExperiment, MzMLFile, MSExperiment
) 
from deepmrm.data.mrm_experiment import MRMExperiment
from deepmrm import data_dir, get_yaml_config

_conf = get_yaml_config()
conf = _conf['PDAC-SIT']

PDAC_SIT_DIR = Path(conf['ROOT_DIR'])
PDAC_SIT_MS_DIR = Path(conf['MZML_DIR'])
PDAC_SIT_MANUAL_DIR = Path(conf['MANUAL_DIR'])
SKYLINE_DIR = Path(conf['SKYLINE_DIR'])
TRANSITION_FILE = conf['TRANSITION_FILE']

def get_msdata_df():
    """Scan file list for PDAC-SIT dataset, and extract metadata from filenames
    Returns:
        pd.DataFrame: file list with metadata
    """
    ms_files = list()
    for fname in PDAC_SIT_MS_DIR.iterdir():
        if fname.is_file():
            file_name = fname.name
            s = file_name.split('_')
            patient_id = int(s[1])
            replicate_id = int(s[2])
            ms_files.append([patient_id, replicate_id, file_name])

    df = pd.DataFrame(ms_files, columns=['patient_id', 'replicate_id', 'mzML'])
    df = df.sort_values(['patient_id', 'replicate_id']).reset_index(drop=True)

    # reset replicate id (one-based index)
    for grp, sub_df in df.groupby('patient_id'):
        df.loc[sub_df.index, 'replicate_id'] = np.arange(1, sub_df.shape[0]+1)

    return df


def get_transition_df():
    """Read a transition list in excel file, and convert it to DataFrame
    Returns:
        [type]: [description]
    """
    tran_df = pd.read_excel(PDAC_SIT_DIR / TRANSITION_FILE)
    tran_df['Heavy'] = tran_df['Compound Name'].str.endswith('.heavy')
    
    tran_df['Sequence'] = tran_df['Compound Name'].replace({
            '\[pS\]': 'S(Phospho)', 
            '\[pT\]': 'T(Phospho)',
            '\(IAA\)': '(Carbamidomethyl)',
            '.heavy': '(Label:13C(6))',
            '.light': ''
            }, regex=True).apply(AASequence.fromString)

    precursor_mass = tran_df['Sequence'].apply(lambda seq: seq.getMonoWeight(Residue.ResidueType.Full, 2))
    tran_df['Precursor Charge'] = (precursor_mass / tran_df['Precursor Ion']).round().astype(int)
    tran_df.index.name = 'compound_idx'

    return tran_df


def get_compact_transition_df():
    
    trans_df = get_transition_df()
    compact_trans_df = trans_df.drop_duplicates(
                        ['Compound Group', 'Compound Name', 'Ret Time (min)'], 
                        keep='first')
    selected_cols = [
        'Compound Group', 'Compound Name', 'Ret Time (min)',
        'Sequence', 'Heavy'
    ]

    compact_trans_df =compact_trans_df[selected_cols].reset_index(drop=True)
    compact_trans_df.index.name = 'compound_idx'

    return compact_trans_df



def extract_manual_data(df, sheet):
    df = df.replace({'X': np.nan, '-': np.nan})
    headers = [header_col[0] for header_col in df.columns]
    ratio_index = headers.index('Final Endo/SIL ratio after inspection')
    
    all_records = []
    for i in range(0, df.shape[0], 3):
        peptide_code = df.iloc[i, 0]
        if pd.isna(peptide_code):
            continue

        cs2_cell_color = sheet[f'D{4+i}'].font.color
        cs3_cell_color = sheet[f'E{4+i}'].font.color
        if (cs2_cell_color and cs2_cell_color.index == 'FFFF0000'):
            selected_charge = 2
        elif (cs3_cell_color and cs3_cell_color.index == 'FFFF0000'):
            selected_charge = 3
        else:
            selected_charge = 0

        trans_quality = [
            sheet[f'R{4+i+k}'].fill.bgColor.value == 64 
                for k in range(2, -1, -1)
        ]

        light_frag_auc = df.iloc[i:i+3, 6].values[::-1]
        # light_frag_rank = df.iloc[i:i+3, 7].values[::-1]
        light_frag_rt = df.iloc[i:i+3, 8].values[::-1]
        light_rt = np.nanmean(light_frag_rt)

        heavy_frag_auc = df.iloc[i:i+3, 12].values[::-1]
        # heavy_frag_rank = df.iloc[i:i+3, 13].values[::-1]
        heavy_frag_rt = df.iloc[i:i+3, 14].values[::-1]
        heavy_rt = np.nanmean(heavy_frag_rt)

        manual_frag_ratio = light_frag_auc / heavy_frag_auc

        light_auc = df.iloc[i, 9]
        heavy_auc = df.iloc[i, 15]
        ion_order = df.iloc[i, 20]
        if not isinstance(ion_order, bool):
            ion_order = ion_order.lower().find('true') > 0

        manual_ratio = df.iloc[i, ratio_index]
        manual_ratio_desc = df.iloc[i, ratio_index+1]
        heavy_pmol = df.iloc[i, ratio_index+2]
        light_pmol = df.iloc[i, ratio_index+3]
        
        record = [
            peptide_code, selected_charge, 
            light_auc, heavy_auc, ion_order, manual_ratio, 
            manual_ratio_desc, light_pmol, heavy_pmol,
            light_rt, heavy_rt,
        ]
        record.extend(list(manual_frag_ratio))
        record.extend(trans_quality)
        record.extend(list(light_frag_auc))
        record.extend(list(heavy_frag_auc))

        all_records.append(record)

    cols = [
        'peptide_code', 'selected_charge', 
        'light_auc', 'heavy_auc', 'ion_order', 
        'manual_ratio', 'manual_ratio_desc', 
        'light_pmol', 'heavy_pmol', 
        'light_rt', 'heavy_rt'
    ]
    cols.extend([f'manual_frag_ratio_t{k+1}' for k in range(3)])
    cols.extend([f'manual_frag_quality_t{k+1}' for k in range(3)])
    cols.extend([f'manual_light_frag_auc_t{k+1}' for k in range(3)])
    cols.extend([f'manual_heavy_frag_auc_t{k+1}' for k in range(3)])

    return pd.DataFrame(all_records, columns=cols)


def create_label_df():
    dfs = []
    for fname in PDAC_SIT_MANUAL_DIR.iterdir():
        if not fname.is_file() or not fname.name.startswith('PDAC'):
            continue
        
        # fname = PDAC_SIT_MANUAL_DIR / 'PDAC273_Sub6_spiked_ratio_rep1-3_outliertest_20211020.xlsx'
        print(fname.name)
        patient_id = int(fname.name[4:7])
        
        wb = load_workbook(fname)
        sheet_names = wb.sheetnames

        for sheet_idx in range(3):
            sh_name = sheet_names[sheet_idx]
            df = pd.read_excel(fname, sheet_name=sheet_idx, header=[0, 1, 2])
            sh = wb[sh_name]
            df = extract_manual_data(df, sh)
            df['patient_id'] = patient_id
            df['replicate_id'] = sheet_idx + 1
            dfs.append(df)

    label_df = pd.concat(dfs, ignore_index=True)

    return label_df

def get_label_df():
    fpath = data_dir / 'PDAC_SIT_label.csv'
    if fpath.exists():
        label_df = pd.read_csv(fpath)
    else:
        label_df = create_label_df()
        # manual correction (Excel file has been updated)
        # m = (label_df['patient_id'] == 211) & (label_df['replicate_id'] == 1) & \
        #     (label_df['peptide_code'] == 'PDAC0097')
        # idx = label_df.index[m][0]
        # label_df.loc[idx, 'manual_ratio'] = (30746+2254)/(313490+43943)

        # m = (label_df['patient_id'] == 186) & (label_df['replicate_id'] == 1) & \
        #     (label_df['peptide_code'] == 'PDAC0168')
        # idx = label_df.index[m][0]        
        # label_df.loc[idx, 'manual_ratio_desc'] = 'y ion transition 순서 불일치'
        label_df.to_csv(fpath, index=False)

    ms_df = get_msdata_df()
    label_df = label_df.merge(ms_df, 
                            on=['patient_id', 'replicate_id'], 
                            how='inner')\
                       .reset_index(drop=True)
    label_df.index.name = 'label_idx'

    return label_df


def create_chrom_df():

    save_path = data_dir / 'PDAC_SIT_chrom.pkl'
    ms_df = get_msdata_df()
    trans_df = get_compact_transition_df()
    sequences = trans_df.loc[:, 'Sequence']
    retention_times = trans_df.loc[:, 'Ret Time (min)']

    all_dfs = []
    for file_id, row in ms_df.iterrows():
        print(f'File index: {file_id}')
        mzml_file = row['mzML']
        patient_id = row['patient_id']
        replicate_id = row['replicate_id']

        mzml_path = PDAC_SIT_MS_DIR / mzml_file
        exp = MRMExperiment()
        mzml = MzMLFile()
        mzml.load(str(mzml_path), exp)
        match_df = exp.get_match_df(sequences=sequences, retention_times=retention_times, mz_tolerance=0.3)

        T = match_df.groupby(['compound_idx'])['chrom_idx'].count()
        if T.shape[0] != trans_df.shape[0]:
            print(f'missing peptides in {mzml_file} ')
            print( np.setdiff1d(trans_df.index, T.index) )
            break

        if np.any(T < 3):
            print(f'missing transitions in {mzml_file} ')
            break

        match_df['patient_id'] = patient_id
        match_df['replicate_id'] = replicate_id
        xics = []
        for chrom_idx in match_df['chrom_idx']:
            x, y = exp.getChromatogram(chrom_idx).get_peaks()
            xics.append(np.stack((x, y)).astype(np.float32))
        match_df['XIC'] = xics
        all_dfs.append(match_df)
    
    all_match_df = pd.concat(all_dfs)
    dtypes_dict = {
        'chrom_idx': np.uint16,
        'mz_error': np.float32,
        'precursor_ion_charge': np.uint8,
        'product_ion_charge': np.uint8,
        'cleavage_index': np.uint8,
        'compound_idx': np.uint16,
        'precursor_mz': np.float32,
        'product_mz': np.float32,
        'min_rt': np.float32,
        'max_rt': np.float32,
        'patient_id': np.uint16,
        'replicate_id': np.uint8,
    }
    all_match_df = all_match_df.astype(dtypes_dict)
    all_match_df.to_pickle(save_path)

    return all_match_df


def get_chrom_df():
    save_path = data_dir / 'PDAC_SIT_chrom.pkl'
    return pd.read_pickle(save_path)


def get_metadata_df():
    
    cpd_df = get_compact_transition_df()
    chrom_df = get_chrom_df()
    label_df = get_label_df()
    sky_df = get_skyline_df()
    # m = sky_df['patient_id'] == 254
    # m &= sky_df['replicate_id'] == 2
    # m &= sky_df['sequence'] == 'TSIVQAAAGGVPGGGSNNGK'
    # sky_df.loc[1644, :]
        
    # 1. Merge skyline result 
    cpd_df['seq_str'] = cpd_df['Sequence'].apply(lambda seq : ''.join([aa.getOneLetterCode() for aa in seq]))
    m = cpd_df['Heavy'] == True
    sky_df = sky_df.merge(cpd_df.loc[m, ['seq_str', 'Compound Group']], 
                how='left',
                left_on='sequence', 
                right_on='seq_str').rename(columns={'Compound Group': 'peptide_code'})
    cols = ['patient_id', 'replicate_id', 'peptide_code', 'start_time', 'end_time']
    label_df = label_df.reset_index().merge(sky_df[cols], 
                how='left',
                on=cols[:3]).set_index('label_idx')
    
    # 2. Add peptide_code, is_heavy in chrom_df 
    chrom_df = chrom_df.merge(
                    # cpd_df[['Compound Group', 'Ret Time (min)', 'Heavy']], 
                    cpd_df[['Compound Group', 'Heavy']],
                    right_index=True, 
                    left_on='compound_idx')

    rename_dict = {'Compound Group': 'peptide_code', 'Heavy': 'is_heavy'}
    chrom_df = chrom_df.rename(columns=rename_dict)

    # 3. Join label_index from label_df 
    key_cols = ['patient_id', 'replicate_id', 'peptide_code']
    chrom_df = chrom_df.merge(
                label_df[key_cols + ['selected_charge']].reset_index(drop=False), 
                left_on=key_cols, 
                right_on=key_cols, 
                how='inner')
    
    # 4. Filter out XICs with selected charges
    m = chrom_df['selected_charge'].notnull()
    m &= chrom_df['precursor_ion_charge'] == chrom_df['selected_charge']
    chrom_df = chrom_df[m]

    cols = [
        'label_idx', 'chrom_idx', 'compound_idx', 'is_heavy',
        'precursor_ion_charge', 'product_ion_charge', 'cleavage_index',
        'precursor_mz', 'product_mz', 'min_rt', 'max_rt',
        'XIC'
    ]

    # 5. data validation
    # 1) three pairs of heavy and light transitions for each pepetide
    sort_cols = key_cols + ['is_heavy', 'cleavage_index']
    chrom_df = chrom_df.sort_values(sort_cols)[cols].set_index('label_idx')

    # for each peptide, there should be 6 transitions
    trans_per_cpd = chrom_df.groupby(['label_idx'])['chrom_idx'].count()
    # removed unpaired transitions (i.e. heavy only or light only)
    label_idx_to_chk = trans_per_cpd[trans_per_cpd != 6].index
    filtered_chrom_df = []
    for label_idx in label_idx_to_chk:
        sub_df = chrom_df.loc[label_idx, :]
        m = sub_df['is_heavy'] == True
        cleavage_index = np.intersect1d(sub_df.loc[m, 'cleavage_index'], sub_df.loc[~m, 'cleavage_index'])
        m = np.in1d(sub_df['cleavage_index'], cleavage_index)
        filtered_chrom_df.append(sub_df[m])

    chrom_df = chrom_df.drop(axis=0, index=label_idx_to_chk)
    chrom_df = pd.concat([chrom_df]+ filtered_chrom_df)

    trans_per_cpd = chrom_df.groupby(['label_idx'])['chrom_idx'].count()
    assert (trans_per_cpd != 6).sum() == 0

    # exclude missing mass-spec data 
    label_df = label_df.loc[chrom_df.index.unique(), :]

    # label = 1 if quantifiable, otherwise 0
    label_df['manual_quality'] = label_df['manual_ratio'].notnull().astype(np.int64)

    good_trans_cnt = label_df['manual_frag_quality_t1'].astype(np.int64) + \
                        label_df['manual_frag_quality_t2'].astype(np.int64) + \
                        label_df['manual_frag_quality_t3'].astype(np.int64)
    m = (good_trans_cnt != 2) & (label_df['manual_quality'] == 1)

    # Quantifiable 경우에 selection 을 하지 않은 경우 모든 transition의 quality는 
    # good으로 간주되어야함
    label_df.loc[m, 'manual_frag_quality_t1'] = True
    label_df.loc[m, 'manual_frag_quality_t2'] = True
    label_df.loc[m, 'manual_frag_quality_t3'] = True

    return label_df, chrom_df


# def extract_chroms():
#     label_df, chrom_df = get_metadata_df()
#     label_df = label_df.sort_values(['patient_id', 'replicate_id'])

#     prev_pid = -1
#     prev_rid = -1
#     cnt = 0
#     for label_idx, row in label_df.iterrows():
#         pid = row['patient_id']
#         rid = row['replicate_id']
#         ms_fname = row['mzML']
#         ms_path = PDAC_SIT_MS_DIR / ms_fname
        
#         c_df = chrom_df.loc[label_idx,  :]
#         if prev_pid != pid or prev_rid != rid:
#             od_exp = OnDiscMSExperiment()
#             _ = od_exp.openFile(str(ms_path))

#         times, xic = [], []
#         for chrom_idx in c_df['chrom_idx']:
#             x, y = od_exp.getChromatogram(chrom_idx).get_peaks()
#             times.append(x)
#             xic.append(y)

#         save_path = f'{PDAC_SIT_TGR_DIR}/{label_idx}.pkl'
#         _ = joblib.dump((times, xic), save_path)
#         prev_pid = pid
#         prev_rid = rid
#         cnt += 1
#         if cnt % 100 == 0:
#             print(cnt)

def get_skyline_df():

    save_path = data_dir / 'PDAC_SIT_skyline.csv'
    def extract_pid_rid(x):
        s = x.split('_')
        pid = s[1]
        rid = s[2]
        if rid == 'Nano':
            rid = s[-2]
        return int(pid), int(rid)

    if save_path.exists():
        df = pd.read_csv(save_path)
    else:
        results = list()
        for sky_path in SKYLINE_DIR.rglob('*.sky'):
            # fname = 'PDAC_Verification_PDAC73_2_Sub4_060821.sky'
            # fname = 'PDAC_Verification_254_Re_Sub6.sky'
            # sky_path = SKYLINE_DIR/fname
            tree = ET.parse(sky_path)
            root = tree.getroot()
            rep_name = root.find('settings_summary').find('measured_results').find('replicate').get('name')

            if sky_path.stem == 'PDAC_153_Rep2_Target_031121_updated':
                pid, rid = 193, 2
            elif sky_path.stem == 'PDAC_153_Rep3_Target_031121_updated':
                pid, rid = 193, 3
            elif sky_path.stem == 'PDAC_Verification_254_Re_Sub6':
                pid, rid = 254, 2
            else:
                pid, rid = extract_pid_rid(rep_name)
            
            for peptide in root.iter('peptide'):
                sequence = peptide.get('sequence')
                precursor = peptide.find('precursor')
                precursor_peak = precursor.find('precursor_results').find('precursor_peak')
                info = {'patient_id': pid, 'replicate_id': rid, 'sequence': sequence}
                info.update(precursor_peak.attrib)
                results.append(info)
        df = pd.DataFrame.from_dict(results)
        df.to_csv(save_path, index=False) 

    float_cols = [
        'peak_count_ratio', 'retention_time', 'start_time',
        'end_time', 'fwhm', 'area', 'background', 'height'
    ]
    for col in float_cols:
        df[col] = df[col].astype(np.float64)
    
    df['patient_id'] = df['patient_id'].astype(int)
    df['replicate_id'] = df['replicate_id'].astype(int)
    df = df.sort_values(['patient_id', 'replicate_id', 'sequence'])

    # reset replicate id (one-based index)
    for grp, sub_df in df.groupby('patient_id'):
        d = {rid: i+1 for i, rid in enumerate(sub_df['replicate_id'].unique())}
        if len(d) != 3:
            ValueError('something wrong')
        df.loc[sub_df.index, 'replicate_id'] = sub_df['replicate_id'].replace(d)

    return df


# df = get_skyline_df()
