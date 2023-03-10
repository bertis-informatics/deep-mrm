import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
from torchvision import transforms as T

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm import get_yaml_config, model_dir
from deepmrm.data.dataset import PRMDataset
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.utils.eval import (
    calculate_area_ratio, create_result_table, 
)
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.transform.make_input import MakeInput
from deepmrm.data_prep import p100_dia, p100_dia_ratio

model_names = [
    'ResNet18_Aug', 
    'ResNet34_Aug', 
    'ResNet50_Aug', 
    'ResNet34_NoAug', 
    'ResNet34_NoAug_NoGrpConv',
    'ResNet34_Aug_NoGrpConv',
]

all_trans_df = p100_dia.get_transition_df()
all_quant_df = p100_dia.get_quant_df()
sample_df = p100_dia.get_sample_df()


label_df = all_quant_df.loc[:, 'Manual'].reset_index(drop=False)

# merge reference RT column
ref_rt_df = all_trans_df[['modified_sequence', 'ref_rt']].drop_duplicates('modified_sequence').reset_index(drop=True)
label_df = label_df.merge(ref_rt_df, on='modified_sequence', how='left')

# evaluation parameters
peptide_id_col = 'modified_sequence'
iou_thresholds = [0.3]
max_detection_thresholds = [1, 3]
mz_tol = Tolerance(20, ToleranceUnit.PPM)
transition_data = TransitionData(all_trans_df, 
                                 peptide_id_col='modified_sequence',
                                 rt_col='ref_rt')
# sample_df = sample_df[sample_df['sample_id'] != 'CC20160706_P100_Plate34_PC3_T3_P-0034_E11_acq_01']


for model_name in model_names:
    
    model_path = Path(model_dir) / f'{model_name}.pth'
    model = torch.load(model_path)
    model = model.set_detections_per_img(128)


    output_dfs = []
    for mzml_idx, row in sample_df.iterrows():
        mzml_path = p100_dia.MZML_DIR / row['mzml_file']
        save_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
        sample_id = row['sample_id']

        m = (label_df['sample_id'] == sample_id) & (label_df['ratio'].notnull()) \
            & (label_df['start_time'].notnull()) & (label_df['end_time'].notnull())

        if ~np.any(m):
            continue
        
        metadata_df = label_df[m]
        
        if 'NoGrpConv' in model_name:
            transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
        else:
            transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False), TransitionDuplicate()])
        
        ds = PRMDataset(mzml_path, transition_data, metadata_df=metadata_df, transform=transform)
        ds.load_data(save_path)

        data_loader = torch.utils.data.DataLoader(
                            ds, batch_size=1, num_workers=8, 
                            collate_fn=obj_detection_collate_fn)
        output_df = model.evaluate(data_loader)

        ds.transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
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
        # output_df.loc[82, :]
        # output_df[output_df['APE_30_det3'].isnull()]
        # idx = 1824
        # output_df.loc[idx, ['start_index', 'end_index']]
        # output_df.loc[idx, 'boxes'][:20]
        # break
        output_dfs.append(output_df)
        print(f'Completed {len(output_dfs)}/{sample_df.shape[0]} files')

    output_dfs = pd.concat(output_dfs, ignore_index=True)

    # output_dfs[output_dfs['APE_30_det3'].isnull()].sort_values('sample_id')
    # m = (output_dfs['dotp_light'] > 0.7)
    # m &= (output_dfs['dotp_heavy'] > 0.7)
    # m = (output_dfs['end_index'] - output_dfs['start_index'] > 5)
    # map_result = calculate_mean_average_precision_recall(
    #                     output_dfs[m], 
    #                     iou_thresholds=iou_thresholds,
    #                     max_detection_thresholds=max_detection_thresholds)
    map_result = None

    save_path = Path(f'reports/DIA_eval_{model_name}.pkl')

    joblib.dump((output_dfs, map_result), save_path)
