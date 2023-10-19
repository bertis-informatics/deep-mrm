import pandas as pd
from pathlib import Path
import numpy as np

import logging
from deepmrm import model_dir
from deepmrm.data.transition import TransitionData
from deepmrm.predict.interface import _load_models, _run_deepmrm
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.utils.plot import plot_heavy_light_pair
from matplotlib import pyplot as plt
from torchvision import transforms as T
from deepmrm.transform.make_xic_input import MakeXicInput
from deepmrm.data.dataset import MRMDataset

# logger = logging.getLogger('deepmrm')


def save_result_images(dataset, result_df, img_save_dir, zoom=True, img_fname=None, score_th=0.5):
    # s = sample_name.split('_')
    # sample_id = s[-3] if s[-1] == 'RR1' else s[-2]
    # img_save_dir = SAVE_DIR / f'{sample_id}'

    ds = dataset
    for idx in range(len(ds)):
        sample = ds[idx]
        row = result_df.iloc[idx]

        time = sample['TIME']/60
        xic = sample['XIC']
        pep_id = sample['peptide_id']

        boxes = row['boxes']
        scores = row['scores']
        rt_boxes = boxes/60
        # box_indexes = np.interp(rt_boxes, time, np.arange(len(time)))

        m = row['scores'] > score_th
        npeaks = np.sum(m)
        colors = ['black', 'blue', 'red']
        
        fig, axs = plot_heavy_light_pair(time, xic)
        if npeaks > 0:
            _ = axs[0].set_title(f'{pep_id}, score: {scores[0]:.2f}')
        else:
            _ = axs[0].set_title(f'{pep_id}, score: 0.0')
        
        for i in range(min(npeaks, len(colors))):
            for k in range(2):
                _ = axs[k].axvline(x=rt_boxes[i, 0], linestyle=':', color=colors[i])
                _ = axs[k].axvline(x=rt_boxes[i, 1], linestyle=':', color=colors[i])

        if zoom and npeaks > 0:
            x_min = rt_boxes[m, 0].min()
            x_max = rt_boxes[m, 1].max()
            plt.xlim([x_min-0.4, x_max+0.4])

        fname = f'{img_fname}.jpg' if img_fname is not None else f'{pep_id}.jpg'
        fig.savefig( img_save_dir / fname )
        plt.close(fig)


def run_deepmrm_with_mzml(model_dir, mzml_path, transition_data, tolerance, transform):
    
    boundary_detector, quality_scorer = _load_models(model_dir)
    ds = MRMDataset(mzml_path, transition_data, transform=transform)
    ds.extract_data(tolerance=tolerance)

    return _run_deepmrm(boundary_detector, quality_scorer, ds, batch_size=8, num_workers=4), ds


PANC_CHK_DIR = Path('/mnt/d/MassSpecData/PancCheck/verification_final')
SAVE_DIR = PANC_CHK_DIR / 'DeepMRM'

def get_transition_df():
    transition_path = PANC_CHK_DIR / 'Transition_Verification_1.xlsx'
    all_trans_df = pd.read_excel(transition_path, sheet_name=0).rename(
                        columns={'Q1': 'precursor_mz', 
                                'Q3': 'product_mz',
                                'ID': 'peptide_id',
                                'Retention time(min)': 'RT'})

    all_trans_df['is_heavy'] = all_trans_df['peptide_id'].apply(lambda x : x.endswith('.heavy'))
    all_trans_df['peptide_id'] = all_trans_df['peptide_id'].replace({'.light': '', '.heavy': ''}, regex=True)
    all_trans_df['RT'] *= 60
    all_trans_df['min_rt'] = all_trans_df['RT'] - 30
    all_trans_df['max_rt'] = all_trans_df['RT'] + 30

    return all_trans_df





SCORE_TH = 0.0
def get_light_heavy_ratio(row):
    if (len(row['scores']) < 1 or row['scores'][0] < SCORE_TH or row['heavy_area'][0] < 1e-4):
        return np.nan
    return row['light_area'][0]/row['heavy_area'][0]

def main():
    
    from deepmrm.constant import LOG_MSG_FORMAT, LOG_DATE_FORMAT
    logging.basicConfig(format=LOG_MSG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.DEBUG)

    mzml_dir = PANC_CHK_DIR / 'mzML'
    mzml_files = list(mzml_dir.glob('./*.mzML'))

    all_trans_df = get_transition_df()
    transition_data = TransitionData(all_trans_df, rt_min_col='min_rt', rt_max_col='max_rt')
    tolerance = Tolerance(1, ToleranceUnit.MZ)


    ratios = []
    scores = []
    rt_seconds = []

    SAVE_DIR.mkdir(exist_ok=True)

    for mzml_path in mzml_files:
        sample_name = mzml_path.stem
        s = sample_name.split('-')
        # sample_id = s[-3] if s[-1] == 'RR1' else s[-2]
        sample_id = s[-1]

        transform = T.Compose([MakeXicInput(
                                cycle_time=0.2,
                                min_rt_key='min_rt', 
                                max_rt_key='max_rt')])
        
        result_dict, ds = run_deepmrm_with_mzml(
                            model_dir, 
                            mzml_path, 
                            transition_data, 
                            tolerance,
                            transform)

        result_df = pd.DataFrame.from_dict(result_dict, orient='index')
        
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

        img_save_dir = SAVE_DIR / f'{sample_id}'
        img_save_dir.mkdir(exist_ok=True)
        save_result_images(ds, result_df, img_save_dir, score_th=0.1)
    
    ratio_df = pd.concat(ratios, axis=1).T
    score_df = pd.concat(scores, axis=1).T
    rt_df = pd.concat(rt_seconds, axis=1).T
    
    ratio_df.index.name = 'sample_id'
    score_df.index.name = 'sample_id'
    rt_df.index.name = 'sample_id'

    ratio_df.to_csv(PANC_CHK_DIR /f'deepmrm_ratio.csv')
    score_df.to_csv(PANC_CHK_DIR /f'deepmrm_score.csv')
    rt_df.to_csv(PANC_CHK_DIR /f'deepmrm_rt.csv')



# manual_df = pd.read_csv(PANC_CHK_DIR /f'manual_ratio.csv', index_col='sample_id')
# ratio_df = pd.read_csv(PANC_CHK_DIR /f'deepmrm_ratio.csv', index_col='sample_id')

# from matplotlib import pyplot as plt

# y0 = manual_df.loc[ratio_df.index, ratio_df.columns].values.reshape(1, -1).flatten()
# y1 = ratio_df.values.reshape(1, -1).flatten()
# s = score_df.values.reshape(1, -1).flatten()

# for col in ratio_df.columns:
#     y0 = manual_df.loc[ratio_df.index, col]
#     y1 = ratio_df.loc[:, col]
#     score = score_df.loc[:, col].mean()

#     plt.figure()
#     plt.scatter(y0, y1, marker='.')
#     plt.xlabel('Manual')
#     plt.ylabel('DeepMRM')
#     plt.title(f'Avg. DeepMRM score: {score:.4f}')
#     plt.savefig(f'./temp/compare/{col}.jpg')


