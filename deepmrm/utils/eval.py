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


def get_ious_from_output_df(row):
    match_quality_matrix = line_iou(
        torch.from_numpy(row['target_boxes']), 
        torch.from_numpy(row['boxes']))
    
    return match_quality_matrix.numpy()

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

        # m = pred_scores > model_score_threshold
        # pred_labels = pred_labels[m]
        # pred_scores = pred_scores[m]
        # pred_boxes = pred_boxes[m]

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
    metric_results['iou_thresholds'] = iou_thresholds

    # update output_df with ious
    ious = metric_results['ious']
    output_df['ious'] = [
            v.reshape((1,)).numpy().astype(np.float32) 
                if v.dim() == 0 else v.squeeze().numpy().astype(np.float32)
                    for v in ious.values()]

    del metric_results['ious']
    
    return metric_results, output_df



def select_xic(peak_quality, quality_threshold):
    auto_selected_xic = np.where(peak_quality > quality_threshold)[0]
    if len(auto_selected_xic) < 2:
        # auto_selected_xic = [pred_quality.argmax()]
        auto_selected_xic = np.arange(len(peak_quality))
    return auto_selected_xic

def calculate_area_ratio(
                    test_ds, 
                    output_df, 
                    iou_threshold=0.5,
                    xic_score_th=0.5):

    """ Estimate the light to heavy ratio for the best matching peak
    """

    auc_results = dict()
    for idx, row in enumerate(output_df.iterrows()):
        index, row = row
        target_boxes = row['target_boxes'][0]
        target_labels = row['target_labels'][0]
        pred_boxes = row['boxes']
        ious = row['ious']
        jj = np.where(ious > iou_threshold)[0]

        if (target_labels < 1) or (len(jj) < 1):
            continue

        j = jj[0] # best scoring peak

        sample = test_ds[idx]
        xic = sample[XIC_KEY]
        time = sample[TIME_KEY]
        
        pred_boxes = pred_boxes[j, :]
        pred_quality = row['peak_quality'][j]

        # from deepmrm.utils.plot import plot_heavy_light_pair
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plot_heavy_light_pair(time, xic)
        # plt.savefig('./temp/temp.jpg')
        
        # convert to integer
        target_boxes = target_boxes.astype(int)
        pred_boxes = pred_boxes.astype(int)

        manually_selected_xic = np.where(sample['manual_peak_quality'] > 0)[0] \
                                    if 'manual_peak_quality' in sample else \
                                np.arange(xic.shape[1])

        auto_selected_xic = select_xic(pred_quality, xic_score_th)
        
        ret = {
                'selected_transition': auto_selected_xic,
                'quality_score': pred_quality[auto_selected_xic].mean(),
                'iou': ious[j],
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

    return quant_df


def create_perf_df(metric_results):

    iou_thresholds = metric_results['iou_thresholds']

    perf_ret = {k: [] for k in ['IoU', 'AP', 'AR1', 'AR3']}
    for th in iou_thresholds:
        perf_ret['IoU'].append(th)
        perf_ret['AP'].append(
            metric_results[f'map_{int(np.around(th*100))}'].item()
        )
        perf_ret['AR1'].append( metric_results[f'mar_{int(np.around(th*100))}_det1'].item() )
        perf_ret['AR3'].append( metric_results[f'mar_{int(np.around(th*100))}_det3'].item() )
        
    return pd.DataFrame.from_dict(perf_ret)


def change_boundary_score_threshold(output_df, score_th):
    
    new_output_df = output_df.copy()
    for idx, row in output_df.iterrows():
        m = row['scores'] > score_th
        for col in ['boxes', 'scores', 'labels', 'peak_quality']:
            if col in new_output_df.columns:
                new_output_df.at[idx, col] = row[col][m]

    return new_output_df




def create_prediction_results(test_ds, output_df, peptide_id_col=None, quality_th=0.5):

    pred_results = {}
    for idx, row in enumerate(output_df.iterrows()):
        index, row = row
        sample = test_ds[idx]
        time = sample[TIME_KEY]
        xic = sample[XIC_KEY]

        pred_boxes = row['boxes']
        pred_scores = row['scores']
        pred_quality = row['peak_quality']

        pred_times = np.interp(pred_boxes.reshape(1, -1), np.arange(len(time)), time)
        pred_times = pred_times.reshape(-1, 2)

        ret = {
            'boxes': pred_times,
            'scores': pred_scores,
            
            'peak_quality': pred_quality,
            'quantification_scores': [],

            'light_area': [],
            'light_background': [],
            'heavy_area': [],
            'heavy_background': [],
            'selected_transition_index': [],

            'light0_area': [],
            'heavy0_area': [],
        }
        
        #for pred_tm, pred_qt in zip(pred_times, pred_quality):
        for pred_box, pred_qt in zip(pred_boxes, pred_quality):
            pred_box = pred_box.astype(np.int32)

            auto_selected_xic = select_xic(pred_qt, quality_th)
            ret['selected_transition_index'].append(auto_selected_xic)
            ret['quantification_scores'].append(pred_qt[auto_selected_xic].mean())

            for i, k in enumerate(['light', 'heavy']):
                summed_xic = xic[i, auto_selected_xic, :].sum(axis=0)
                peak_area, background = calculate_peak_area(
                                            time, summed_xic, 
                                            pred_box[0], pred_box[1])
                ret[f'{k}_area'].append(peak_area)
                ret[f'{k}_background'].append(background)
            
                summed_xic = xic[i, :, :].sum(axis=0)
                peak_area, background = calculate_peak_area(
                                            time, summed_xic, 
                                            pred_box[0], pred_box[1])
                ret[f'{k}0_area'].append(peak_area)

        if peptide_id_col is not None:
            ret[peptide_id_col] = sample[peptide_id_col]

        pred_results[index] = ret

    return pred_results