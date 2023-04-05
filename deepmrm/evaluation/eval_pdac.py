import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from mstorch.data.manager import DataManager
from mstorch.enums import PartitionType
from deepmrm import model_dir, private_project_dir
from deepmrm.data_prep import get_metadata_df
from deepmrm.data.dataset import DeepMrmDataset
from deepmrm.utils.eval import (
    calculate_area_ratio, 
    compute_peak_detection_performance,
    create_perf_df,
)
from deepmrm.data import obj_detection_collate_fn
from deepmrm.train.train_boundary_detector import RANDOM_SEED, task, transform
from deepmrm.predict.interface import _load_models


reports_dir = private_project_dir / 'reports'
batch_size = 128
num_workers = 8
dataset_name = 'PDAC'

save_path = reports_dir/f'{dataset_name}_output_df.pkl'
label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=False)

ds = DeepMrmDataset(
            label_df, 
            pdac_chrom_df,
            scl_chrom_df,
            transform=transform)

data_mgr = DataManager(
                task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn,
                random_seed=RANDOM_SEED)
data_mgr.split()

testset_loader = data_mgr.get_dataloader(PartitionType.TEST, batch_size=batch_size)

if not save_path.exists():
    model, model_qs = _load_models(model_dir)

    output_df = model.predict(testset_loader)
    output_df = model_qs.predict(testset_loader.dataset, output_df)
    output_df.to_pickle(save_path)
else:
    output_df = pd.read_pickle(save_path)

iou_thresholds = list(np.arange(1, 10)*0.1)


metric_results, output_df = compute_peak_detection_performance(
                    output_df, 
                    max_detection_thresholds=[1, 3],
                    iou_thresholds=iou_thresholds)

     
perf_df = create_perf_df(metric_results)
perf_df

output_df['ious']

# precisions = metric_results['precisions']['30_det1']
# recalls = metric_results['recalls']['30_det1']
# m = recalls > -1
# from matplotlib import pyplot as plt
# plt.figure()
# plt.rcParams.update({'font.size': 13})
# plt.plot(recalls[m], precisions[m])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.savefig(reports_dir / f'figures/{dataset_name}_pr_curve.jpg')



#####################################################################
###### AR, AP metrics at different IoUs
test_ds = testset_loader.dataset
iou_threshold = 0.3

# Quantification performance 
quant_df = calculate_area_ratio(
                test_ds, 
                output_df, 
                iou_threshold=iou_threshold, 
                xic_score_th=0.5)
cols = [
        'selected_transition', 'quality_score', 
        'manual_ratio', 'pred_ratio', 'pred0_ratio', 'iou'
    ]
output_df = output_df.drop(columns=['manual_ratio'])
output_df = output_df.join(quant_df[cols], how='left')



quant_df = output_df
# import joblib
# joblib.dump((output_df, metric_results), save_path)

y_true = quant_df.loc[:, 'manual_ratio'] 
y_pred = quant_df.loc[:, 'pred_ratio'] 
m = y_true.notnull()
y_true = y_true[m]
y_pred = y_pred[m]
iou = quant_df.loc[m, 'iou']
ape = np.abs(y_true - y_pred)/y_true
aape = np.arctan(ape)

iou_effect = []
for iou_cutoff in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    m = (iou_cutoff <= iou) & (iou < iou_cutoff+0.1)
    iou_effect.append([np.sum(m),
    ape[m].mean(),
    aape[m].mean(),
    pearsonr(y_true[m], y_pred[m])[0],
    spearmanr(y_true[m], y_pred[m])[0],])

pd.DataFrame(iou_effect)


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

           
del label_df, pdac_chrom_df, scl_chrom_df