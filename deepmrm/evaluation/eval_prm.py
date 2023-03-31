import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
from torchvision import transforms as T

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.transform.make_input import SelectSegment, MakeInput, MakeTagets
from deepmrm import model_dir, private_project_dir
from deepmrm.data.dataset import PRMDataset
from deepmrm.data.transition import TransitionData
from deepmrm.data import obj_detection_collate_fn
from deepmrm.data_prep import p100_prm
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
batch_size = 1
num_workers = 1
dataset_name = 'PRM'
xic_dir = p100_prm.XIC_DIR
meta_df, trans_df = p100_prm.get_metadata_df()

mzml_files = meta_df['mzml_file'].unique()

peptide_id_col = 'modified_sequence'
iou_thresholds = [0.3]
max_detection_thresholds = [1, 3]
dotp_threshold = 0.8
mz_tol = Tolerance(20, ToleranceUnit.PPM)
transition_data = TransitionData(trans_df, rt_col='ref_rt', peptide_id_col='modified_sequence')
model, model_qs = _load_models(model_dir)

m = meta_df['is_heavy']
dotp_heavy = meta_df.loc[m, ['sample_id', peptide_id_col, 'dotp']].copy()
dotp_light = meta_df.loc[~m, ['sample_id', peptide_id_col, 'dotp']].copy()
dotp_heavy.columns = ['sample_id', peptide_id_col, 'dotp_heavy']
dotp_light.columns = ['sample_id', peptide_id_col, 'dotp_light']
meta_df = meta_df.merge(dotp_heavy, on=['sample_id', peptide_id_col], how='left')\
                 .merge(dotp_light, on=['sample_id', peptide_id_col], how='left')

transform = T.Compose([SelectSegment(), MakeInput(use_rt=False), MakeTagets()])
save_path = Path(reports_dir/f'{dataset_name}_output_df.pkl')


if not save_path.exists():
    # task = model.task
    output_dfs = []
    for mzml_idx, mzml_name in enumerate(mzml_files):
        
        xic_path = xic_dir / (mzml_name[:-4] + 'pkl')
        mzml_path = p100_prm.MZML_DIR/mzml_name
        m = meta_df['mzml_file'] == mzml_name
        metadata_df = meta_df[m]
        
        # dotp score merge
        s_df = metadata_df.groupby(peptide_id_col)['start_time'].min()
        e_df = metadata_df.groupby(peptide_id_col)['end_time'].max()
        metadata_df = metadata_df.drop(columns=['start_time', 'end_time'])\
                            .merge(s_df, left_on=peptide_id_col, right_index=True)\
                            .merge(e_df, left_on=peptide_id_col, right_index=True)
        metadata_df = metadata_df.drop_duplicates([peptide_id_col]).drop(columns=['is_heavy'])
        metadata_df['manual_quality'] = 1
        metadata_df['manual_boundary'] = 1

        ds = PRMDataset(
                    mzml_path, 
                    transition_data, 
                    metadata_df=metadata_df,
                    transform=transform)
        ds.load_data(xic_path)

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


iou_thresholds = list(np.arange(1, 10)*0.1)
# new_output_df = change_boundary_score_threshold(output_dfs, score_th=0.1)
gt = (output_dfs['dotp_heavy'] > 0.7) & (output_dfs['dotp_light'] > 0.7)
new_output_df = output_dfs[gt]

metric_results, output_df = compute_peak_detection_performance(
                    new_output_df, 
                    max_detection_thresholds=[1, 3],
                    iou_thresholds=iou_thresholds)

     
perf_df = create_perf_df(metric_results)
perf_df

# m = output_dfs['pred_ratio'].isnull() & output_dfs['pred0_ratio'].notnull() 
# output_dfs.loc[m, 'pred_ratio'] = output_dfs.loc[m, 'pred0_ratio']
y_true = output_dfs.loc[:, 'manual_ratio'] 
y_pred = output_dfs.loc[:, 'pred_ratio'] 
m = (output_dfs['dotp_heavy'] > 0.7) & (output_dfs['dotp_light'] > 0.7)
m &= output_dfs.loc[:, 'manual_ratio'].notnull()

y_true = y_true[m]
y_pred = y_pred[m]
ape = np.abs(y_true - y_pred)/(y_true)
aape = np.arctan(ape)

ret = dict()
ret['PCC'] = pearsonr(y_true, y_pred)[0]
ret['SPC'] = spearmanr(y_true, y_pred)[0]
# exclude zero values for MAPE
m = y_true > 1e-6
ret['MAPE'] = ape[m].mean()
ret['MdAPE'] = ape.median()
ret['MAAPE'] = aape.mean()

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
