import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as T
import joblib

from deepmrm import get_yaml_config, model_dir
from deepmrm.data.dataset import MRMDataset
from deepmrm.data_prep import eoc
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.constant import RT_KEY, XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.utils.eval import (
    calculate_area_ratio, create_result_table, 
    calculate_mean_average_precision_recall
)
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn

mzml_dir = eoc.MZML_DIR
meta_df, trans_df = eoc.get_metadata_df()
mzml_files = meta_df['mzml_file'].unique()

peptide_id_col = 'modified_sequence'

model_names = [
    'ResNet18_Aug', 
    'ResNet34_Aug', 
    'ResNet50_Aug', 
    'ResNet34_NoAug', 
    'ResNet34_NoAug_NoGrpConv',
    'ResNet34_Aug_NoGrpConv',
]

peptide_id_col = 'modified_sequence'
iou_thresholds = [0.3]
max_detection_thresholds = [1, 3]
mz_tol = Tolerance(0.5, ToleranceUnit.MZ)


for model_name in model_names:
    
    model_path = Path(model_dir) / f'{model_name}.pth'
    model = torch.load(model_path)
    model = model.set_detections_per_img(20)
    task = model.task

    if 'NoGrpConv' in model_name:
        transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
    else:
        transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False), TransitionDuplicate()])

    output_dfs = []

    for mzml_idx, mzml_name in enumerate(mzml_files):
        mzml_path = mzml_dir / mzml_name
        if not mzml_path.exists():
            continue

        print(mzml_path)
        m = meta_df['mzml_file'] == mzml_name
        #label_df = meta_df[m].drop_duplicates(['protein_name', peptide_id_col])
        tmp_df = meta_df.loc[m, [peptide_id_col, 'RT']].groupby(peptide_id_col)['RT'].count() 
        tmp_df = tmp_df[tmp_df == 2]
        tmp_df.name = 'cnt'

        label_df = meta_df[m].merge(tmp_df, left_on=peptide_id_col, right_index=True, how='inner')
        label_df = label_df.iloc[:, :-1]
        label_df = label_df.drop_duplicates([peptide_id_col])

        m = np.in1d(trans_df[peptide_id_col], label_df[peptide_id_col].unique())
        transition_data = TransitionData(trans_df[m], rt_col='ref_rt')

        ds = MRMDataset(mzml_path, transition_data, metadata_df=label_df, transform=transform)
        ds.extract_data(tolerance=mz_tol)

        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, collate_fn=obj_detection_collate_fn)
        output_df = model.evaluate(data_loader)

        ds.transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
        output_df = create_result_table(ds, output_df[['boxes', 'scores', 'labels']])
        # cols = [peptide_id_col, 'skyline_name', 'mzml_file']
        # output_df = output_df.merge(ds.metadata_df[cols], on=peptide_id_col)
        
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
                # output_df = output_df.join(quant_df['APE'].rename(f'APE_{int(iou_th*100)}_det{max_det}'), how='left')

        output_dfs.append(output_df)    
        print(f'Completed {mzml_idx+1}/{len(mzml_files)} files')        

    output_dfs = pd.concat(output_dfs, ignore_index=True)
    gt_mask = output_dfs['start_time'].notnull()
    # 잘린 XIC는 제외???
    # gt_mask &= output_dfs['start_time'] - output_dfs['xic_start'] > 10
    # gt_mask &= output_dfs['xic_end'] - output_dfs['end_time'] > 10
    map_result = calculate_mean_average_precision_recall(
                        output_dfs[gt_mask], 
                        iou_thresholds=iou_thresholds,
                        max_detection_thresholds=max_detection_thresholds)
    
    save_path = Path(f'reports/MRM_eval_{model_name}.pkl')
    joblib.dump((output_dfs, map_result), save_path)    


# all_result_df = pd.concat(dfs, ignore_index=True)
# all_result_df.to_pickle('reports/MRM_results_trim.pkl')
# all_result_df.to_csv('reports/MRM_results_trim.csv', index=False)

# for i, k in enumerate(['manual', 'pred']):
#     all_result_df[f'{k}_ratio'] = all_result_df[f'{k}_light_area']/all_result_df[f'{k}_heavy_area']

# y_true = all_result_df['manual_ratio']
# y_pred = all_result_df['pred_ratio']

# all_result_df['APE'] = np.abs(y_true - y_pred)/y_true
# m = (y_true < 1e-2) & (y_pred < 1e-2)
# all_result_df.loc[m, 'APE'] = 0

# gt_mask = all_result_df['start_time'].notnull()

# pred_mask = (all_result_df['pred_match_idx'] > -1) & (all_result_df['pred_match_idx'] < 3)
# pred_mask &= all_result_df['iou'] > 0.1
# pred_mask &= all_result_df['pred_quality'] > 0
# pred_mask &= gt_mask

# np.sum(pred_mask)/np.sum(gt_mask)

# all_result_df.loc[pred_mask, 'APE'].mean()
# all_result_df.loc[pred_mask, 'APE'].median()