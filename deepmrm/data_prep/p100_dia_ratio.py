import pandas as pd
import numpy as np
from torchvision import transforms as T

from deepmrm import data_dir, private_data_dir
from deepmrm.data_prep import p100_dia
from deepmrm.data.transition import TransitionData
from deepmrm.transform.make_input import MakeInput
from deepmrm.data.dataset import PRMDataset
from deepmrm.constant import RT_KEY, XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.utils.peak import calculate_peak_area_with_time


def get_quantitative_transition_df(label_type):
    csv_path = p100_dia.SKY_DIR /label_type /'Transition Results.csv'
    df = pd.read_csv(csv_path)

    seq = pd.Series(df['Peptide Modified Sequence'].unique())
    seq.name = 'Peptide Modified Sequence'
    norm_seq = p100_dia.normalize_seq_str(seq)
    norm_seq.name = 'modified_sequence'
    seq = seq.to_frame().join(norm_seq)
    df = df.merge(seq, on='Peptide Modified Sequence')
    return df


def compute_manual_ratio(label_type):
    """
    Calculate peak area using manually annotated boundary and transitions
    """
    assert label_type in ['Manual', 'Skyline', 'AvG']

    all_trans_df = p100_dia.get_transition_df()
    all_quant_df = p100_dia.get_quant_df()
    sample_df = p100_dia.get_sample_df()
    label_df = all_quant_df.loc[:, label_type].reset_index(drop=False)

    q_trans_df = get_quantitative_transition_df(label_type)
    transition_data = TransitionData(all_trans_df, 
                                     peptide_id_col='modified_sequence',
                                     rt_col='ref_rt')

    auc_results = []
    for mzml_idx, row in sample_df.iterrows():

        mzml_path = p100_dia.MZML_DIR / row['mzml_file']
        save_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
        sample_id = row['sample_id']
        print(mzml_path)

        m = (label_df['sample_id'] == sample_id) & (label_df['ratio'].notnull()) \
            & (label_df['start_time'].notnull()) & (label_df['end_time'].notnull())

        if not np.any(m):
            continue
        
        metadata_df = label_df[m]
        transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
        
        ds = PRMDataset(mzml_path, transition_data, metadata_df=metadata_df, transform=transform)
        ds.load_data(save_path)

        for idx in range(len(ds)):
            sample = ds[idx]
            xic = sample[XIC_KEY]
            time = sample[TIME_KEY]
            rt = sample[RT_KEY]
            seq = sample['modified_sequence']
            st_time, ed_time = sample['start_time'], sample['end_time']

            m = (transition_data.df['modified_sequence'] == seq) & (transition_data.df['is_heavy'])
            t_df = transition_data.df.loc[m, ['product_ion', 'product_charge']].reset_index(drop=True)
            
            m = (q_trans_df['Replicate'] == sample_id) & (q_trans_df['Quantitative']) &\
                (q_trans_df['modified_sequence'] == seq)
            selected_trans_df = q_trans_df[m].drop_duplicates(['Fragment Ion', 'Product Charge'], keep='first')

            # selected_trans_df[['Fragment Ion', 'Product Charge']]
            t_df = t_df.reset_index(drop=False)
            t_df = t_df.merge(selected_trans_df, 
                        left_on=['product_ion', 'product_charge'], 
                        right_on=['Fragment Ion', 'Product Charge'], 
                        how='inner')
            index = t_df['index'].values
            
            ret = {'sample_id': sample_id, 'modified_sequence': seq}
            for i, k in enumerate(['light', 'heavy']):
                summed_xic = xic[i, index, :].sum(axis=0)
                peak_area, background = calculate_peak_area_with_time(
                                            time, summed_xic, st_time, ed_time)
                ret[f'manual_{k}_area'] = peak_area
                ret[f'manual_{k}_background'] = background
            auc_results.append(ret)
    ratio_df = pd.DataFrame(auc_results)
    return ratio_df


def compute_area_ratio():
    df1 = compute_manual_ratio('Manual')
    df2 = compute_manual_ratio('AvG')
    df2.columns = [col.replace('manual', 'AvG') for col in df2.columns]
    df = df1.merge(df2, on=['sample_id', 'modified_sequence'])
    # df.to_csv(data_dir / 'p100_dia_ratio.csv', index=False)
    return df


def get_manual_ratio_df():
    save_path = data_dir / 'P100_DIA_ratio.csv'
    if not save_path.exists():
        df = compute_area_ratio()
        df.to_csv(save_path, index=False)
    else:
        df = pd.read_csv(save_path)
    
    return df