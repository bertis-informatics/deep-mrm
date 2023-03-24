import pandas as pd
import numpy as np
import torch

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm import model_dir, private_project_dir
from deepmrm.data.dataset import PRMDataset
from deepmrm.data.transition import TransitionData
from deepmrm.data_prep import p100_dia_ratio, p100_dia
from deepmrm.predict.interface import _load_models
from deepmrm.data import obj_detection_collate_fn
from deepmrm.utils.eval import (
    calculate_area_ratio, 
    compute_peak_detection_performance,
    create_perf_df,
    change_boundary_score_threshold,
    get_ious_from_output_df
)

reports_dir = private_project_dir / 'reports'
batch_size = 128
num_workers = 8
dataset_name = 'DIA'

all_trans_df = p100_dia.get_transition_df()
all_quant_df = p100_dia.get_quant_df()
all_sample_df = p100_dia.get_sample_df()

# T = all_quant_df.loc[:, ('AvG', 'end_time')] - all_quant_df.loc[:, ('AvG', 'start_time')]
# ep = T[T.notnull()]
# ep.mean()
# ep.std()

label_df = all_quant_df.loc[:, 'Manual'].reset_index(drop=False)

# merge reference RT column
ref_rt_df = all_trans_df[['modified_sequence', 'ref_rt']].drop_duplicates('modified_sequence').reset_index(drop=True)
label_df = label_df.merge(ref_rt_df, on='modified_sequence', how='left')

# evaluation parameters
peptide_id_col = 'modified_sequence'
iou_thresholds = list(np.arange(1, 10)*0.1)
max_detection_thresholds = [1, 3]
mz_tol = Tolerance(20, ToleranceUnit.PPM)
transition_data = TransitionData(
                        all_trans_df, 
                        peptide_id_col='modified_sequence',
                        rt_col='ref_rt')

model, model_qs = _load_models(model_dir)

save_path = reports_dir/f'{dataset_name}_output_df.pkl'

sample_df = all_sample_df.sample(10)
# model.detector.score_thresh = 1e-3
# sample_df = all_sample_df

output_dfs = []
for mzml_idx, row in sample_df.iterrows():
    mzml_path = p100_dia.MZML_DIR / row['mzml_file']
    xic_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
    sample_id = row['sample_id']

    m = (label_df['sample_id'] == sample_id) & (label_df['ratio'].notnull()) \
        & (label_df['start_time'].notnull()) & (label_df['end_time'].notnull())

    if not np.any(m):
        continue
    
    metadata_df = label_df[m].copy()
    metadata_df['manual_quality'] = 1
    ds = PRMDataset(mzml_path, 
                    transition_data, 
                    metadata_df=metadata_df, 
                    transform=model.transform)
    ds.load_data(xic_path)
    data_loader = torch.utils.data.DataLoader(
                                    ds, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    collate_fn=obj_detection_collate_fn)
    
    output_df = model.predict(data_loader)
    # output_df = model_qs.predict(data_loader.dataset, output_df)
    # output_df['ious'] = output_df.apply(get_ious_from_output_df, axis=1)
    # quant_df = calculate_area_ratio(ds, output_df, iou_threshold=0.3)
    # cols = [
    #         'selected_transition', 'quality_score', 
    #         'manual_ratio', 'pred_ratio', 'pred0_ratio']
    # output_df = output_df.join(quant_df[cols], how='left')
    # # here, the manual ratio is calculated with all transitions
    # output_df = output_df.rename(columns={'manual_ratio': 'manual0_ratio'})

    output_dfs.append(output_df)
    print(f'Completed {len(output_dfs)}/{sample_df.shape[0]} files')

output_dfs = pd.concat(output_dfs, ignore_index=True)


iou_thresholds = list(np.arange(1, 10)*0.1)
# new_output_df = change_boundary_score_threshold(output_dfs, score_th=0.05)
new_output_df = output_dfs
# new_output_df[['boxes', 'scores', 'target_boxes']]

fig3_df = p100_dia.load_figure3_data()
ratio_df = p100_dia_ratio.get_manual_ratio_df()

## 'AAPEAS(Phospho)SPPASPLQHLLPGK' was exlucded in AvG analysis
col1 = ['modified_sequence', 'sample_id']
ratio_df = ratio_df.merge(fig3_df[col1], on=col1, how='inner')

ratio_df['manual_ratio'] = ratio_df['manual_light_area']/ratio_df['manual_heavy_area']
ratio_df['AvG_ratio'] = ratio_df['AvG_light_area']/ratio_df['AvG_heavy_area']
ratio_df.loc[ratio_df['manual_heavy_area']==0, 'manual_ratio'] = np.nan

# ratio_df['manual_ratio'].isnull().sum()
# ratio_df['AvG_ratio'].isnull().sum()

cols = ['sample_id', 'modified_sequence', 'manual_ratio', 'AvG_ratio']
new_output_df = new_output_df.merge(ratio_df[cols], on=cols[:-2], how='inner')

gt = new_output_df['manual_ratio'].notnull()
gt &= new_output_df['AvG_ratio'].notnull()


metric_results, output_df = compute_peak_detection_performance(
                                    new_output_df[gt], 
                                    max_detection_thresholds=[1, 3],
                                    iou_thresholds=iou_thresholds)


perf_df = create_perf_df(metric_results)
perf_df




#     ds.transform = T.Compose([MakeInput(ref_rt_key='ref_rt', use_rt=False)])
#     output_df = create_result_table(ds, output_df[['boxes', 'scores', 'labels']])

#     for iou_th in iou_thresholds:
#         for max_det in max_detection_thresholds:
#             quant_df = calculate_area_ratio(ds, output_df, iou_threshold=iou_th, max_det_threshold=max_det, score_th=0.1)

#             col_rename = {
#                 'pred_ratio': f'pred_ratio_{int(iou_th*100)}_det{max_det}',
#                 'manual_ratio': f'manual_ratio_{int(iou_th*100)}_det{max_det}',
#                 'APE': f'APE_{int(iou_th*100)}_det{max_det}'
#             }
#             quant_df = quant_df.rename(columns=col_rename)
#             output_df = output_df.join(quant_df[list(col_rename.values())], how='left')
#     # output_df.loc[82, :]
#     # output_df[output_df['APE_30_det3'].isnull()]
#     # idx = 1824
#     # output_df.loc[idx, ['start_index', 'end_index']]
#     # output_df.loc[idx, 'boxes'][:20]
#     # break
#
# output_dfs = pd.concat(output_dfs, ignore_index=True)


# # output_dfs[output_dfs['APE_30_det3'].isnull()].sort_values('sample_id')
# # m = (output_dfs['dotp_light'] > 0.7)
# # m &= (output_dfs['dotp_heavy'] > 0.7)
# # m = (output_dfs['end_index'] - output_dfs['start_index'] > 5)
# # map_result = calculate_mean_average_precision_recall(
# #                     output_dfs[m], 
# #                     iou_thresholds=iou_thresholds,
# #                     max_detection_thresholds=max_detection_thresholds)
# map_result = None

# save_path = Path(f'reports/DIA_eval_{model_name}.pkl')

# joblib.dump((output_dfs, map_result), save_path)
