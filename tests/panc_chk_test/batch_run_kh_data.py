# import sys
# sys.path.append('/home/jungkap/workspace/disease-classification')
import pandas as pd
from pathlib import Path
import torch
import numpy as np

from deepmrm import model_dir
from deepmrm.data.transition import TransitionData
from deepmrm.predict.interface import _load_models, _run_deepmrm
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.utils.plot import plot_heavy_light_pair
from matplotlib import pyplot as plt
from torchvision import transforms as T
from deepmrm.transform.transition import SingleTransition
from deepmrm.transform.make_xic_input import MakeXicInput
from panc_chk_test.batch_run import run_deepmrm_with_mzml, save_result_images

proj_root = Path('/mnt/c/Users/jungk/OneDrive - Bertis Inc/Documents/Projects/PancCheck')
proj_dir = proj_root / '2023_KH_data'
proj_ms_dir = Path('/mnt/d/MassSpecData/PancCheck/kh_data')


def get_sample_mapping_table():
    map_df = pd.read_excel(proj_dir / 'sample_name_mapping.xlsx', skiprows=0, header=1)
    map_df = map_df.rename(columns={'Sample Name': 'sample_id'})
    map_df['set_index'] = map_df['set'].apply(lambda x : int(x[-1]))
    cols = ['No', 'set_index', 'sample_id','wiff_name', 'mzML', 'wiff_name.1', 'mzML.1', 'wiff_name.2', 'mzML.2']
    map_df = map_df[cols]
    map_df['rep1_mzml'] =  map_df['wiff_name'] + '-' + map_df['mzML'] + '.mzml'
    map_df['rep2_mzml'] =  map_df['wiff_name.1'] + '-' + map_df['mzML.1'] + '.mzml'
    map_df['rep3_mzml'] =  map_df['wiff_name.2'] + '-' + map_df['mzML.2'] + '.mzml'

    map_df = map_df[['No', 'set_index', 'sample_id','rep1_mzml', 'rep2_mzml', 'rep3_mzml']]
    return map_df


def get_kh_transition_df():
    transition_path = proj_dir / 'panc check transitions (13-candidates).xlsx'

    all_trans_df = pd.read_excel(transition_path, sheet_name=0, 
                                 skiprows=0, header=1).rename(
                        columns={'Q1': 'precursor_mz', 
                                'Q3': 'product_mz',
                                'name': 'peptide_id',
                                #'Retention time(min)': 'RT',
                                }).drop(columns=['Unnamed: 0'])
    
    m = all_trans_df['peptide_id'].str.startswith('C5.NADYSYSVWK')
    all_trans_df = all_trans_df[~m].reset_index()
    all_trans_df['is_heavy'] = all_trans_df['peptide_id'].apply(lambda x : x.endswith('.heavy'))
    all_trans_df['peptide_id'] = all_trans_df['peptide_id'].replace({'.light': '', '.heavy': ''}, regex=True)
    
    for col in ['min_rt_10', 'max_rt_10', 'min_rt_12', 'max_rt_12']:
        all_trans_df[col] *= 60
    
    return all_trans_df


def get_mzml_files(rep_id):
    map_df = get_sample_mapping_table()
    i = rep_id
    return [proj_ms_dir / f'Test{i}/mzml' / fname for fname in map_df[f'rep{i}_mzml']]


SCORE_TH = 0.0
def get_light_heavy_ratio(row):
    if (len(row['scores']) < 1 or row['scores'][0] < SCORE_TH or row['heavy_area'][0] < 1e-4):
        return np.nan
    return row['light_area'][0]/row['heavy_area'][0]




def quant_kh_data(rep_id, cycle_time=0.1):

    # rep_id = 1
    map_df = get_sample_mapping_table()
    all_trans_df = get_kh_transition_df()
    gradient = 10 if rep_id == 1 else 12

    # i = 2
    # st = i*2
    # all_trans_df = all_trans_df.iloc[st:st+2, :].copy()
    # all_trans_df = all_trans_df.iloc[18:20, :].copy()
    rt_min_col, max_rt_col = f'min_rt_{gradient}', f'max_rt_{gradient}'
    # rt_min_col, max_rt_col = None, None

    transition_data = TransitionData(all_trans_df, rt_min_col=rt_min_col, rt_max_col=max_rt_col)
    tolerance = Tolerance(1, ToleranceUnit.MZ)

    save_dir = proj_ms_dir / f'Test{rep_id}/DeepMRM' 
    save_dir.mkdir(exist_ok=True)

    ratios = []
    scores = []
    rt_seconds = []
    for idx, row in map_df.iterrows():
        # ii = np.where(map_df['sample_id'] == 'PC_267')[0][0]
        # row = map_df.iloc[ii]
        # row = map_df.iloc[np.random.randint(0, map_df.shape[0])]
        # transition_data.df['min_rt_10'] = 1*60
        # transition_data.df['max_rt_10'] = 3.5*60 
        # transition_data.df['min_rt_12'] = 2*60
        # transition_data.df['max_rt_12'] = 3.5*60 

        sample_id = row['sample_id']
        mzml_path = proj_ms_dir / f'Test{rep_id}/mzml' / row[f'rep{rep_id}_mzml']
        transform = T.Compose([MakeXicInput(
                                cycle_time=cycle_time,
                                min_rt_key=rt_min_col, 
                                max_rt_key=max_rt_col),
                                # SingleTransition(),
                            ])
        result_dict, ds = run_deepmrm_with_mzml(
                            model_dir, 
                            mzml_path, 
                            transition_data, 
                            tolerance,
                            transform)

        result_df = pd.DataFrame.from_dict(result_dict, orient='index')
        
        # img_save_dir = save_dir / f'{sample_id}'
        # img_save_dir.mkdir(exist_ok=True)
        # img_save_dir = Path('/home/jungkap/workspace/auto-mrm/temp')
        # save_result_images(ds, result_df, img_save_dir, zoom=False, img_fname='xic_temp', score_th=0.2)

        ratio = result_df.apply(get_light_heavy_ratio, axis=1)
        ratio.index = transition_data.get_peptide_ids()
        ratio.name = sample_id
        
        score = result_df['scores'].apply(lambda x : x[0] if len(x) > 0 else 0)
        score.index = ratio.index
        score.name = sample_id

        rt = result_df['boxes'].apply(lambda x : x[0].mean() if len(x) > 0 else np.nan)
        rt.index = ratio.index
        rt.name = sample_id

        ratios.append(ratio)
        scores.append(score)
        rt_seconds.append(rt)
        
    ratio_df = pd.concat(ratios, axis=1).T
    score_df = pd.concat(scores, axis=1).T
    rt_df = pd.concat(rt_seconds, axis=1).T
    
    ratio_df.to_csv(proj_dir /f'replicate-{rep_id}-deepmrm_ratio.csv')
    score_df.to_csv(proj_dir /f'replicate-{rep_id}-deepmrm_score.csv')
    rt_df.to_csv(proj_dir /f'replicate-{rep_id}-deepmrm_rt.csv')


if __name__ == "__main__":
    quant_kh_data(rep_id=3)

# plt.figure()
# score_df.boxplot()
# plt.xticks([])
# plt.savefig(save_dir/'deepmrm_score.jpg')



