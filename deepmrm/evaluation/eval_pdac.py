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
    change_boundary_score_threshold
)
from deepmrm.data import obj_detection_collate_fn
from deepmrm.train.train_boundary_detector import RANDOM_SEED, task, transform
from deepmrm.predict.interface import _load_models


reports_dir = private_project_dir / 'reports'
batch_size = 128
num_workers = 8
dataset_name = 'PDAC'

save_path = reports_dir/f'{dataset_name}_output_df.pkl'
label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=False, only_quantifiable_peak=True)

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
new_output_df = change_boundary_score_threshold(output_df, score_th=0.05)
metric_results, output_df = compute_peak_detection_performance(
                    new_output_df, 
                    max_detection_thresholds=[1, 3],
                    iou_thresholds=iou_thresholds)

     
perf_df = create_perf_df(metric_results)
perf_df

# joblib.dump((output_df, map_result), save_path)


#####################################################################
###### AR, AP metrics at different IoUs
test_ds = testset_loader.dataset
xic_score_th = 0.5

# Quantification performance 
quant_df = calculate_area_ratio(test_ds, output_df, iou_threshold=0.5, xic_score_th=0.5)

y_true = quant_df.loc[:, 'manual_ratio'] 
y_pred = quant_df.loc[:, 'pred_ratio'] 

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



# for th in iou_thresholds:
#     # m = (quant_df['iou'] >= th) & (quant_df['iou'] < th+0.1)
#     m = (quant_df['iou'] > th)
#     maape = quant_df.loc[m, 'AAPE'].mean()
#     mape = quant_df.loc[m, 'APE'].mean()
#     pcc = pearsonr(y_true[m], y_pred[m])[0]
#     spc = spearmanr(y_true[m], y_pred[m])[0]
#     perf_ret['MAPE'].append(mape)
#     perf_ret['MAAPE'].append(maape)
#     perf_ret['PCC'].append(pcc)
#     perf_ret['SPC'].append(spc)
# perf_df
           
del label_df, pdac_chrom_df, scl_chrom_df