import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
from torchvision import transforms as T

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm import model_dir
from deepmrm.data.dataset import PRMDataset
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.data_prep import p100_prm
from scipy.stats import pearsonr, spearmanr
from deepmrm.utils.eval import (
    calculate_area_ratio, 
    compute_peak_detection_performance,
    create_perf_df,
    change_boundary_score_threshold
)

xic_dir = p100_prm.XIC_DIR
meta_df, trans_df = p100_prm.get_metadata_df()

mzml_files = meta_df['mzml_file'].unique()

peptide_id_col = 'modified_sequence'
iou_thresholds = [0.3]
max_detection_thresholds = [1, 3]
dotp_threshold = 0.8
mz_tol = Tolerance(20, ToleranceUnit.PPM)
transition_data = TransitionData(trans_df, rt_col='ref_rt')

m = meta_df['is_heavy']
dotp_heavy = meta_df.loc[m, ['sample_id', peptide_id_col, 'dotp']].copy()
dotp_light = meta_df.loc[~m, ['sample_id', peptide_id_col, 'dotp']].copy()
dotp_heavy.columns = ['sample_id', peptide_id_col, 'dotp_heavy']
dotp_light.columns = ['sample_id', peptide_id_col, 'dotp_light']
meta_df = meta_df.merge(dotp_heavy, on=['sample_id', peptide_id_col], how='left')\
                 .merge(dotp_light, on=['sample_id', peptide_id_col], how='left')


for model_name in model_names:
   
    model_path = Path(model_dir) / f'{model_name}.pth'
    model = torch.load(model_path)
    model = model.set_detections_per_img(20)
    task = model.task
    output_dfs = []

    for mzml_idx, mzml_name in enumerate(mzml_files):
        
        xic_path = xic_dir / (mzml_name[:-4] + 'pkl')
        mzml_path = p100_prm.MZML_DIR/mzml_name
        m = meta_df['mzml_file'] == mzml_name
        metadata_df = meta_df[m]
        
        if 'NoGrpConv' in model_name:
            transform = T.Compose([
                SelectSegment(),
                MakeInput(use_rt=False)])
        else:
            transform = T.Compose([
                SelectSegment(),
                MakeInput(use_rt=False), 
                TransitionDuplicate()])
        
        # dotp score merge
        s_df = metadata_df.groupby(peptide_id_col)['start_time'].min()
        e_df = metadata_df.groupby(peptide_id_col)['end_time'].max()
        metadata_df = metadata_df.drop(columns=['start_time', 'end_time'])\
                            .merge(s_df, left_on=peptide_id_col, right_index=True)\
                            .merge(e_df, left_on=peptide_id_col, right_index=True)
        metadata_df = metadata_df.drop_duplicates([peptide_id_col]).drop(columns=['is_heavy'])

        ds = PRMDataset(
                    mzml_path, 
                    transition_data, 
                    metadata_df=metadata_df,
                    transform=transform)
        ds.load_data(xic_path)

        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=8, collate_fn=obj_detection_collate_fn)
        output_df = model.evaluate(data_loader)
        ds.transform = T.Compose([SelectSegment(), MakeInput(ref_rt_key='ref_rt', use_rt=False)])
        output_df = create_result_table(ds, output_df[['boxes', 'scores', 'labels']])

        for iou_th in iou_thresholds:
            for max_det in max_detection_thresholds:
                quant_df = calculate_area_ratio(ds, output_df, iou_threshold=iou_th, max_det_threshold=max_det, score_th=0.1)
                col_rename = {
                    'pred_ratio': f'pred_ratio_{int(iou_th*100)}_det{max_det}',
                    'manual_ratio': f'manual_ratio_{int(iou_th*100)}_det{max_det}',
                    'APE': f'APE_{int(iou_th*100)}_det{max_det}'
                }
                quant_df = quant_df.rename(columns=col_rename)
                output_df = output_df.join(quant_df[list(col_rename.values())], how='left')
        
        output_dfs.append(output_df)
        print(f'Completed {mzml_idx+1}/{len(mzml_files)} files')
    
    output_dfs = pd.concat(output_dfs, ignore_index=True)
    map_result = None
    
    save_path = Path(f'reports/PRM_eval_{model_name}.pkl')
    joblib.dump((output_dfs, map_result), save_path)

