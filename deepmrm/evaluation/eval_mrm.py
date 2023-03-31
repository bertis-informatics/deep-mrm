import pandas as pd
import numpy as np
from pathlib import Path
import torch

from deepmrm import model_dir, private_project_dir
from deepmrm.data.dataset import MRMDataset
from deepmrm.data_prep import eoc
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.predict.interface import _load_models
from scipy.stats import pearsonr, spearmanr
from deepmrm.utils.eval import (
    calculate_area_ratio, 
    compute_peak_detection_performance,
    create_perf_df,
    change_boundary_score_threshold,
    get_ious_from_output_df
)

reports_dir = private_project_dir / 'reports'
batch_size = 32
num_workers = 4
dataset_name = 'MRM'

model, model_qs = _load_models(model_dir)

mzml_dir = eoc.MZML_DIR
meta_df, trans_df = eoc.get_metadata_df()
mzml_files = meta_df['mzml_file'].unique()

peptide_id_col = 'modified_sequence'
mz_tol = Tolerance(0.5, ToleranceUnit.MZ)

save_path = Path(reports_dir/f'{dataset_name}_output_df.pkl')

if not save_path.exists():
    output_dfs = []
    for mzml_idx, mzml_name in enumerate(mzml_files):
        mzml_path = mzml_dir / mzml_name
        if not mzml_path.exists():
            continue

        print(mzml_path)
        m = meta_df['mzml_file'] == mzml_name
        tmp_df = meta_df.loc[m, [peptide_id_col, 'RT']].groupby(peptide_id_col)['RT'].count() 
        tmp_df = tmp_df[tmp_df == 2]
        tmp_df.name = 'cnt'

        label_df = meta_df[m].merge(tmp_df, left_on=peptide_id_col, right_index=True, how='inner')
        label_df = label_df.iloc[:, :-1]
        label_df = label_df.drop_duplicates([peptide_id_col])
        label_df['manual_quality'] = 1
        label_df['manual_boundary'] = 1

        m = np.in1d(trans_df[peptide_id_col], label_df[peptide_id_col].unique())
        transition_data = TransitionData(
                                    trans_df[m], 
                                    rt_col='ref_rt', 
                                    peptide_id_col=peptide_id_col)

        ds = MRMDataset(
                    mzml_path, 
                    transition_data, 
                    metadata_df=label_df, 
                    transform=model.transform)
        ds.extract_data(tolerance=mz_tol)

        data_loader = torch.utils.data.DataLoader(ds, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    collate_fn=obj_detection_collate_fn)
        output_df = model.predict(data_loader)
        output_df = model_qs.predict(data_loader.dataset, output_df)

        output_df['ious'] = output_df.apply(get_ious_from_output_df, axis=1)
        quant_df = calculate_area_ratio(ds, output_df, iou_threshold=0.3)
        cols = [
            'selected_transition', 'quality_score', 
            'manual_ratio', 'pred_ratio', 'pred0_ratio']
        
        output_df = output_df.join(quant_df[cols], how='left')
        output_dfs.append(output_df)    
        print(f'Completed {mzml_idx+1}/{len(mzml_files)} files')
    
    output_dfs = pd.concat(output_dfs, ignore_index=True)
    output_dfs.to_pickle(save_path)
else:
    output_dfs = pd.read_pickle(save_path)

    
# gt_mask = output_dfs['start_time'].notnull()
# ### exclude truncated XICs?? 
# gt_mask &= output_dfs['start_time'] - output_dfs['xic_start'] > 10
# gt_mask &= output_dfs['xic_end'] - output_dfs['end_time'] > 10

iou_thresholds = list(np.arange(1, 10)*0.1)

metric_results, output_df = compute_peak_detection_performance(
                    output_dfs, 
                    max_detection_thresholds=[1, 3],
                    iou_thresholds=iou_thresholds)
     
perf_df = create_perf_df(metric_results)
perf_df


# output_dfs['target_bo']
quant_df = output_dfs
y_true = quant_df.loc[:, 'manual_ratio'] 
y_pred = quant_df.loc[:, 'pred_ratio'] 
m = y_true.notnull()
y_true = y_true[m]
y_pred = y_pred[m]

quant_df['APE'] = np.abs(y_true - y_pred)/y_true
quant_df['AAPE'] = np.arctan(quant_df['APE'])

ret = dict()
ret['PCC'] = pearsonr(y_true, y_pred)[0]
ret['SPC'] = spearmanr(y_true, y_pred)[0]
ret['MAPE'] = quant_df.loc[:, 'APE'].mean()
ret['MdAPE'] = quant_df.loc[:, 'APE'].median()
ret['MAAPE'] = quant_df.loc[:, 'AAPE'].mean()

print(ret)
pd.DataFrame.from_dict(ret, orient='index').T


# saving for scatter plot
save_path = reports_dir / f'{dataset_name}_ratio_df.pkl'
y_true.to_frame().join(y_pred).to_pickle(save_path)


# save for PR-curve
pr_df = pd.DataFrame.from_dict({
            'precision-det1': metric_results['precisions']['30_det1'].numpy(),
            'recalls-det1': metric_results['recalls']['30_det1'].numpy(),
            'precision-det3': metric_results['precisions']['30_det3'].numpy(),
            'recalls-det3': metric_results['recalls']['30_det3'].numpy(),            
        })
save_path = reports_dir / f'{dataset_name}_pr_df.pkl'
pr_df.to_pickle(save_path)

