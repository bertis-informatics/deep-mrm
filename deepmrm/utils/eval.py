from pathlib import Path
import pandas as pd
import numpy as np
import torch
from mstorch.utils.logger import get_logger
 
from deepmrm.constant import RT_KEY, XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.utils.peak import calculate_peak_area
from deepmrm.model.utils import line_iou
from deepmrm.utils.mean_ap import MeanAveragePrecisionRecall
from deepmrm.utils.transition_select import find_best_transition_index

fig_dir = Path('./reports/figures')
logger = get_logger()


def compute_peak_detection_performance(
                output_df, 
                max_detection_thresholds=[1, 3],
                iou_thresholds=list(np.arange(1, 10)*0.1)):
    
    metric = MeanAveragePrecisionRecall(
                iou_thresholds=iou_thresholds,
                max_detection_thresholds=max_detection_thresholds)

    preds, target = [], []
    for index, row in output_df.iterrows():
        pred_labels = row['labels']
        pred_scores = row['scores']
        pred_boxes = row['boxes']

        target_boxes = row['target_boxes']
        target_labels = row['target_labels']
        
        z = np.zeros((pred_boxes.shape[0], 1))
        pred_boxes = np.concatenate(
                        (pred_boxes[:, :1], z, pred_boxes[:, 1:], z+1), axis=1)
        preds.append({
            'boxes': torch.from_numpy(pred_boxes),
            'scores': torch.from_numpy(pred_scores),
            'labels': torch.from_numpy(pred_labels)
        })
        
        z = np.zeros((target_boxes.shape[0], 1))
        # manual_label = row['manual_quality'] if 'manual_quality' in row else 1
        # if manual_label == 0:
        #     target.append({ 
        #             'boxes': torch.zeros((0, 4), dtype=torch.float64), 
        #             'labels': torch.zeros((0), dtype=torch.int64)})
        # else:
        target_boxes = np.concatenate(
                            (target_boxes[:, :1], z, target_boxes[:, 1:], z+1), 
                            axis=1)
        # manual_box = torch.FloatTensor([[row['start_index'], 0, row['end_index'], 1]])
        target.append({ 
                'boxes': torch.from_numpy(target_boxes), 
                'labels': torch.from_numpy(target_labels)
            })
    metric.update(preds, target)
    
    metric_results = metric.compute()
    metric_results['recall_thresholds'] = np.array(metric.rec_thresholds, dtype=np.float32)

    # update output_df with ious
    ious = metric_results['ious']
    output_df['ious'] = [
            v.reshape((1,)).numpy().astype(np.float32) 
                if v.dim() == 0 else v.squeeze().numpy().astype(np.float32)
                    for v in ious.values()
        ]
    del metric_results['ious']
    
    return metric_results, output_df



def calculate_area_ratio(test_ds, output_df, xic_score_th=0.5):

    """ Estimate the light to heavy ratio for the best matching peak
    """

    auc_results = dict()
    for idx, row in enumerate(output_df.iterrows()):
        index, row = row
        target_boxes = row['target_boxes'][0]
        target_labels = row['target_labels'][0]
        pred_boxes = row['boxes']
        
        if (target_labels < 1) or (len(pred_boxes) < 1):
            continue

        sample = test_ds[idx]
        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        pred_boxes = pred_boxes[0, :]
        pred_quality = row['peak_quality'][0]

        # from deepmrm.utils.plot import plot_heavy_light_pair
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plot_heavy_light_pair(time, xic)
        # plt.savefig('./temp/temp.jpg')
        
        # convert to integer
        target_boxes = target_boxes.astype(int)
        pred_boxes = pred_boxes.astype(int)
        manually_selected_xic = np.where(sample['manual_peak_quality'] > 0)[0]

        auto_selected_xic = np.where(pred_quality > xic_score_th)[0]
        if len(auto_selected_xic) < 1:
            auto_selected_xic = [pred_quality.argmax()]

        ret = {
                'selected_transition': auto_selected_xic,
                'quality_score': pred_quality[auto_selected_xic].mean()
            }
        for i, k in enumerate(['light', 'heavy']):
            summed_xic = xic[i, manually_selected_xic, :].sum(axis=0)
            
            peak_area, background = calculate_peak_area(time, summed_xic, target_boxes[0], target_boxes[1])
            ret[f'manual_{k}_area'] = peak_area
            ret[f'manual_{k}_background'] = background
            
            if len(auto_selected_xic) > 0:
                summed_xic = xic[i, auto_selected_xic, :].sum(axis=0)
                peak_area, background = calculate_peak_area(time, summed_xic, pred_boxes[0], pred_boxes[1])
            else:
                peak_area, background = np.nan, np.nan
            ret[f'pred_{k}_area'] = peak_area
            ret[f'pred_{k}_background'] = background

            if len(auto_selected_xic) != xic.shape[1]:
                summed_xic = xic[i, :, :].sum(axis=0)
                peak_area, background = calculate_peak_area(time, summed_xic, pred_boxes[0], pred_boxes[1])
            ret[f'pred0_{k}_area'] = peak_area
            ret[f'pred0_{k}_background'] = background

        auc_results[index] = ret

    quant_df = pd.DataFrame.from_dict(auc_results, orient='index')
    for i, k in enumerate(['manual', 'pred', 'pred0']):
        quant_df[f'{k}_ratio'] = quant_df[f'{k}_light_area']/quant_df[f'{k}_heavy_area']
    # y_true = quant_df['manual_ratio']
    # y_pred = quant_df['pred_ratio']
    # quant_df['APE'] = np.abs(y_true - y_pred)/y_true    

    # m = quant_df['pred_ratio'].isnull()
    # m &= quant_df['manual_ratio'] > 1
    # quant_df[m]

    return quant_df

# quant_df = quant_df.join(output_df[['manual_peak_quality', 'ious', 'manual_ratio']],
#                         rsuffix='_ori')

# iou_th = 0.3
# m = quant_df['ious'].apply(lambda x : x[0] > iou_th)

# np.corrcoef(quant_df.loc[m, 'manual_ratio'], quant_df.loc[m, 'pred_ratio'])
# np.corrcoef(quant_df.loc[m, 'manual_ratio'], quant_df.loc[m, 'pred0_ratio'])

# # np.corrcoef(quant_df.loc[m, 'manual_ratio'], quant_df.loc[m, 'manual_ratio_ori'])
# y_true = quant_df.loc[:, 'manual_ratio_ori']
# y_pred = quant_df.loc[:, 'manual_ratio']
# ape = np.abs(y_true - y_pred)/y_true    

# quant_df[ape > 3]

# aape = np.arctan(ape)
# maape = np.mean(aape)
# mape = ape.mean()
# print(maape)
# print(mape)

# m2 = m & (ape > 0.2)
# quant_df[m2]

# output_df.loc[4982, :]


# # quant_df[['manual_peak_quality','selected_transition']]
# T =  quant_df['manual_peak_quality'].apply(lambda x: x.sum())
# m = T < 2
# quant_df.loc[m, ['manual_peak_quality','selected_transition']]

# plt.scatter(quant_df['manual_ratio'], quant_df['pred_ratio'])
# def compute_peak_detection_performance(
#                         test_ds, 
#                         output_df, 
#                         iou_thresholds,
#                         max_detection_thresholds):
    
#     # calculate AR, mAP and then update output_df with IoUs
#     map_result, output_df = calculate_mean_average_precision_recall(
#                                 output_df, 
#                                 iou_thresholds=iou_thresholds,
#                                 max_detection_thresholds=max_detection_thresholds
#                             )

#     # for iou_th in iou_thresholds:
#     #     for max_det in max_detection_thresholds:
#     #         quant_df = calculate_area_ratio(
#     #                         test_ds,
#     #                         output_df,
#     #                         iou_threshold=iou_th,
#     #                         max_det_threshold=max_det)
#     #                         # score_th=score_th)
#     #         col_rename = {
#     #             'selected_xic_pair': f'selected_xic_pair_{int(iou_th*100)}_det{max_det}',
#     #             'pred_ratio': f'pred_ratio_{int(iou_th*100)}_det{max_det}',
#     #             'manual_ratio': f'manual_ratio_{int(iou_th*100)}_det{max_det}',
#     #             'APE': f'APE_{int(iou_th*100)}_det{max_det}',
#     #         }
#     #         quant_df = quant_df.rename(columns=col_rename)
#     #         output_df = output_df.join(quant_df[list(col_rename.values())], how='left')
    
#     return output_df, map_result


# result_df = pd.read_pickle('reports/PDAC_results.pkl')

# def create_result_table(test_ds, output_df):

#     meta_df = test_ds.metadata_df
#     output_df = meta_df.join(output_df)
    
#     manual_bondary_index = {}
#     for idx, row in enumerate(output_df.iterrows()):
#         index, row = row
#         sample = test_ds[idx]
#         xic = sample[XIC_KEY]
#         time = sample[TIME_KEY]
#         st = sample['start_time']
#         ed = sample['end_time']
#         rt = sample[RT_KEY]
#         xic_cnt = xic.shape[1]
#         manual_bondary_idx = np.interp([st, rt, ed], time, np.arange(len(time)))
#         manual_bondary_index[index] = {
#             'start_index': manual_bondary_idx[0],
#             'rt_index': manual_bondary_idx[1],
#             'end_index': manual_bondary_idx[2],
#             'xic_count': xic_cnt,
#             'xic_start': time[0],
#             'xic_end': time[-1],
#         }
    
#     output_df = output_df.join(
#             pd.DataFrame.from_dict(manual_bondary_index, orient='index'))
            
#     return output_df
