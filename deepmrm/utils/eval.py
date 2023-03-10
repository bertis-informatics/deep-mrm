from pathlib import Path
import pandas as pd
import numpy as np
import torch
from mstorch.utils.logger import get_logger
 
from deepmrm.constant import RT_KEY, XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.utils.peak import calculate_peak_area_with_time, calculate_peak_area_with_index
from deepmrm.model.utils import line_iou
from deepmrm.utils.mean_ap import MeanAveragePrecisionRecall
from deepmrm.utils.transition_select import find_best_transition_index

fig_dir = Path('./reports/figures')
logger = get_logger()

# result_df = pd.read_pickle('reports/PDAC_results.pkl')


def create_result_table(test_ds, output_df):

    meta_df = test_ds.metadata_df
    output_df = meta_df.join(output_df)
    
    manual_bondary_index = {}
    for idx, row in enumerate(output_df.iterrows()):
        index, row = row
        sample = test_ds[idx]
        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        st = sample['start_time']
        ed = sample['end_time']
        rt = sample[RT_KEY]
        xic_cnt = xic.shape[1]
        manual_bondary_idx = np.interp([st, rt, ed], time, np.arange(len(time)))
        manual_bondary_index[index] = {
            'start_index': manual_bondary_idx[0],
            'rt_index': manual_bondary_idx[1],
            'end_index': manual_bondary_idx[2],
            'xic_count': xic_cnt,
            'xic_start': time[0],
            'xic_end': time[-1],
        }
    
    output_df = output_df.join(
            pd.DataFrame.from_dict(manual_bondary_index, orient='index'))
            
    return output_df


def calculate_mean_average_precision_recall(
            output_df, 
            score_th=0.05,
            max_detection_thresholds=[1, 3, 5],
            iou_thresholds=[0.2, 0.3, 0.5]):
    
    metric = MeanAveragePrecisionRecall(
                iou_thresholds=iou_thresholds,
                max_detection_thresholds=max_detection_thresholds)

    chk_mq = 'manual_quality' in output_df.columns

    for index, row in output_df.iterrows():
        pred_labels = row['labels']
        pred_scores = row['scores']
        pred_boxes = row['boxes']

        preds, target = [], []
        if chk_mq:
            quantifiable = row['manual_quality'] == 1
            manual_label = row['manual_quality']
        else:
            quantifiable = True
            manual_label = 1
            
        # m = (pred_labels == 2) & (pred_scores > score_th)
        m = (pred_scores > score_th)
        pred_labels = pred_labels[m]
        pred_boxes = pred_boxes[m]
        pred_scores = pred_scores[m]

        # indexes = get_filter_index(pred_labels, pred_scores)
        z = np.zeros((pred_boxes.shape[0], 1))
        pred_boxes = np.concatenate(
                        (pred_boxes[:, :1], z, pred_boxes[:, 1:], z+1), axis=1)
        preds.append({
            'boxes': torch.from_numpy(pred_boxes),
            'scores': torch.from_numpy(pred_scores),
            'labels': torch.from_numpy(pred_labels)
        })

        if manual_label > 0:
            manual_box = torch.FloatTensor([[row['start_index'], 0, row['end_index'], 1]])
            target.append({ 
                'boxes': manual_box, 
                'labels': torch.tensor([manual_label])})    
        else:
            target.append({ 
                'boxes': torch.zeros((0, 4), dtype=torch.float64), 
                'labels': torch.zeros((0), dtype=torch.int64)})    
        
        # target.append({ 
        #     'boxes': manual_box, 
        #     'labels': torch.tensor([manual_label])
        # })

        # if quantifiable:
        #     target.append({ 'boxes': manual_box, 'labels': torch.tensor([2])})
        # else:
        #     target.append({ 'boxes': torch.tensor([]), 'labels': torch.tensor([])})

        metric.update(preds, target)
    
    metric_results = metric.compute()
    metric_results = {k: v.item() for k, v in metric_results.items()}

    return metric_results

def calculate_area_ratio(test_ds, output_df, 
                iou_threshold, 
                max_det_threshold,
                score_th):
    """ Estimate the light to heavy ratio for the best matching peak
    """

    auc_results = dict()

    for idx, row in enumerate(output_df.iterrows()):
        
        index, row = row
        pred_boxes = row['boxes'].copy()

        if 'manual_quality' in row and row['manual_quality'] == 0:
            continue

        sample = test_ds[idx]
        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        rt = sample[RT_KEY]
        st, ed = row['start_index'], row['end_index']
        st_time, ed_time = row['start_time'], row['end_time']
        
        is_pdac = 'replicate_id' in sample and sample['replicate_id'] > 0

        pred_boxes = row['boxes'].copy()
        pred_scores = row['scores'].copy()
        
        # m = pred_labels == 2
        m = pred_scores > score_th
        pred_boxes = pred_boxes[m]
        pred_scores = pred_scores[m]

        if len(pred_boxes) < 1:
            continue

        pred_boxes = pred_boxes[:max_det_threshold]
        pred_scores = pred_scores[:max_det_threshold]
        
        manual_boxes = torch.from_numpy(np.array([st, ed]).reshape(1, -1)) 
        iou = line_iou(manual_boxes, torch.from_numpy(pred_boxes))
        iou = iou[0, :].numpy()
        m = (iou > iou_threshold)
        
        best_match_idx = -1
        if np.any(m):
            best_match_idx = np.where(m)[0][0]
            iou_score = iou[best_match_idx]
            
            pred_bondary_idx = pred_boxes[best_match_idx]
            pred_bondary_time = np.interp(pred_bondary_idx, np.arange(len(time)), time)
            pred_st_time, pred_ed_time = pred_bondary_time
            
            if is_pdac:
                manually_selected_xic = np.array([
                    f_ for f_ in range(3) if sample[f'manual_frag_quality_t{f_+1}']])
            else:
                manually_selected_xic = np.arange(xic.shape[1])
            
            auto_selected_xic = find_best_transition_index(time, xic, pred_st_time, pred_ed_time)
            # from mstorch.utils.visualize.xic import create_peak_group_image
            # img = create_peak_group_image(time=time, xic_tensor=xic, boundary=[pred_st_time, pred_ed_time])
            # img.save('tmp.jpg')
            ret = {
                'pred_match_idx': best_match_idx,
                'iou_score': iou_score,
                'selected_xic_pair': auto_selected_xic
            }            

            for i, k in enumerate(['light', 'heavy']):
                summed_xic = xic[i, manually_selected_xic, :].sum(axis=0)
                peak_area, background = calculate_peak_area_with_time(time, summed_xic, st_time, ed_time)
                ret[f'manual_{k}_area'] = peak_area
                ret[f'manual_{k}_background'] = background
                if best_match_idx > -1:
                    summed_xic = xic[i, auto_selected_xic, :].sum(axis=0)
                    peak_area, background = calculate_peak_area_with_time(time, summed_xic, pred_st_time, pred_ed_time)
                else:
                    peak_area, background = 0, 0
                ret[f'pred_{k}_area'] = peak_area
                ret[f'pred_{k}_background'] = background

            auc_results[index] = ret

    if len(auc_results) < 1:
        return pd.DataFrame([], columns=[
                'pred_match_idx', 'iou_score', 'selected_xic_pair',
                'manual_light_area', 'manual_light_background',
                'manual_heavy_area', 'manual_heavy_background',
                'manual_ratio', 'pred_ratio', 'APE'])

    quant_df = pd.DataFrame.from_dict(auc_results, orient='index')
    for i, k in enumerate(['manual', 'pred']):
        quant_df[f'{k}_ratio'] = quant_df[f'{k}_light_area']/quant_df[f'{k}_heavy_area']
    y_true = quant_df['manual_ratio']
    y_pred = quant_df['pred_ratio']
    quant_df['APE'] = np.abs(y_true - y_pred)/y_true    
    # quant_df.loc[(y_true < 1e-2) & (y_pred < 1e-2), 'APE'] = 0

    return quant_df


def compute_peak_detection_performance(test_ds, output_df, 
                    iou_thresholds=[0.3, 0.5], 
                    max_detection_thresholds=[1, 3],
                    score_th=0.05):

    output_df = create_result_table(test_ds, output_df)
    for iou_th in iou_thresholds:
        for max_det in max_detection_thresholds:

            quant_df = calculate_area_ratio(
                            test_ds, 
                            output_df, 
                            iou_threshold=iou_th, 
                            max_det_threshold=max_det,
                            score_th=score_th)
            col_rename = {
                'selected_xic_pair': f'selected_xic_pair_{int(iou_th*100)}_det{max_det}',
                'pred_ratio': f'pred_ratio_{int(iou_th*100)}_det{max_det}',
                'manual_ratio': f'manual_ratio_{int(iou_th*100)}_det{max_det}',
                'APE': f'APE_{int(iou_th*100)}_det{max_det}',
            }
            quant_df = quant_df.rename(columns=col_rename)
            output_df = output_df.join(quant_df[list(col_rename.values())], how='left')
    
    map_result = calculate_mean_average_precision_recall(
                        output_df, 
                        score_th=score_th,
                        iou_thresholds=iou_thresholds,
                        max_detection_thresholds=max_detection_thresholds)

    return output_df, map_result