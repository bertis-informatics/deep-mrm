import time
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
    compute_peak_detection_performance,
    create_perf_df,
    get_ious_from_output_df,
    calculate_area_ratio
)

reports_dir = private_project_dir / 'reports'
batch_size = 1
num_workers = 1
dataset_name = 'DIA'

all_trans_df = p100_dia.get_transition_df()
all_quant_df = p100_dia.get_quant_df()
all_sample_df = p100_dia.get_sample_df()

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

is_decoy = False

if is_decoy:
    save_path = reports_dir/f'{dataset_name}_output_df_decoy.pkl'
else:
    save_path = reports_dir/f'{dataset_name}_output_df.pkl'

sample_df = all_sample_df.sample(10, random_state=2023)
# sample_df = all_sample_df

if not save_path.exists():
    st_tm = time.time()
    output_dfs = []
    for mzml_idx, row in sample_df.iterrows():
        mzml_path = p100_dia.MZML_DIR / row['mzml_file']
        if is_decoy:
            xic_path = p100_dia.XIC_DIR / f'{mzml_path.stem}_decoy.pkl'
        else:
            xic_path = p100_dia.XIC_DIR / f'{mzml_path.stem}.pkl'
        
        sample_id = row['sample_id']
        m = (label_df['sample_id'] == sample_id) & (label_df['ratio'].notnull()) \
            & (label_df['start_time'].notnull()) & (label_df['end_time'].notnull())

        if not np.any(m):
            continue
        
        metadata_df = label_df[m].copy()
        metadata_df['manual_quality'] = 1
        metadata_df['manual_boundary'] = 1
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
        
        output_df = model_qs.predict(data_loader.dataset, output_df)
        output_df['ious'] = output_df.apply(get_ious_from_output_df, axis=1)
        quant_df = calculate_area_ratio(ds, output_df, iou_threshold=0.3)
        cols = [
                'selected_transition', 'quality_score', 
                'manual_ratio', 'pred_ratio', 'pred0_ratio']
        output_df = output_df.join(quant_df[cols], how='left')
        # here, the manual ratio is calculated with all transitions
        output_df = output_df.rename(columns={'manual_ratio': 'manual0_ratio'})

        output_dfs.append(output_df)
        print(f'Completed {len(output_dfs)}/{sample_df.shape[0]} files')

    output_dfs = pd.concat(output_dfs, ignore_index=True)
    ed_tm = time.time()
    (ed_tm-st_tm)/sample_df.shape[0]
    (ed_tm-st_tm)/output_dfs.shape[0]
    
    output_dfs.to_pickle(save_path)
else:
    output_dfs = pd.read_pickle(save_path)



iou_thresholds = list(np.arange(1, 10)*0.1)
new_output_df = output_dfs

fig3_df = p100_dia.load_figure3_data()
ratio_df = p100_dia_ratio.get_manual_ratio_df()

## 'AAPEAS(Phospho)SPPASPLQHLLPGK' was exlucded in AvG analysis
col1 = ['modified_sequence', 'sample_id']
ratio_df = ratio_df.merge(fig3_df[col1], on=col1, how='inner')

ratio_df['manual_ratio'] = ratio_df['manual_light_area']/ratio_df['manual_heavy_area']
ratio_df['AvG_ratio'] = ratio_df['AvG_light_area']/ratio_df['AvG_heavy_area']
ratio_df.loc[ratio_df['manual_heavy_area']==0, 'manual_ratio'] = np.nan

cols = ['sample_id', 'modified_sequence', 'manual_ratio', 'AvG_ratio']
new_output_df = new_output_df.merge(ratio_df[cols], on=cols[:-2], how='inner')

# new_output_df[['manual_ratio', 'manual_ratio_re']]
gt = new_output_df['manual_ratio'].notnull()
gt &= new_output_df['AvG_ratio'].notnull()
metric_results, output_df = compute_peak_detection_performance(
                                    new_output_df[gt],
                                    max_detection_thresholds=[1, 3],
                                    iou_thresholds=iou_thresholds)

perf_df = create_perf_df(metric_results)
perf_df

from scipy.stats import pearsonr, spearmanr
gt = new_output_df['manual_ratio'].notnull()
gt &= new_output_df['AvG_ratio'].notnull()
gt &= new_output_df['pred_ratio'].notnull()
gt &= (new_output_df['pred_ratio'] != np.inf)

y_true = new_output_df.loc[gt, 'manual_ratio']
y_pred = new_output_df.loc[gt, 'pred_ratio']

ape = np.abs(y_true - y_pred)/y_true
aape = np.arctan(ape)

ret = dict()
ret['PCC'] = pearsonr(y_true, y_pred)[0]
ret['SPC'] = spearmanr(y_true, y_pred)[0]
ret['MAPE'] = ape.mean()
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





