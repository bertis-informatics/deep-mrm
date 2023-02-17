import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as T

from mstorch.utils import get_logger
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.constant import TIME_KEY, XIC_KEY
from deepmrm.utils.peak import calculate_peak_area_with_time
from deepmrm.utils.transition_select import find_best_transition_index
from deepmrm.data.transition import TransitionData
from deepmrm.data.dataset import MRMDataset
from deepmrm.data import obj_detection_collate_fn
from deepmrm.predict.input_xic import XicData, DeepMrmInputDataset

logger = get_logger('DeepMRM')
device = torch.device('cpu')

def create_prediction_results(test_ds, output_df, peptide_id_col=None, score_th=0.1):

    # reset transform
    original_transforms = T.Compose(list(test_ds.transform.transforms))
    new_transforms = T.Compose([
        t for t in test_ds.transform.transforms if not isinstance(t, TransitionDuplicate)])
    
    test_ds.set_transform(new_transforms)

    pred_results = {}
    for idx, row in enumerate(output_df.iterrows()):
        
        index, row = row
        sample = test_ds[idx]
        time = sample[TIME_KEY]
        xic = sample[XIC_KEY]
        pred_labels = row['labels']
        pred_boxes = row['boxes']
        pred_scores = row['scores']
        
        # filtering by score cutoff
        # m = pred_labels == 1
        m = pred_scores > score_th
        pred_boxes = pred_boxes[m]
        pred_scores = pred_scores[m]
        pred_labels = pred_labels[m]

        pred_times = np.interp(pred_boxes.reshape(1, -1), np.arange(len(time)), time)
        pred_times = pred_times.reshape(-1, 2)

        ret = {
            'boxes': pred_times,
            'scores': pred_scores,
            'light_area': [],
            'light_background': [],
            'heavy_area': [],
            'heavy_background': [],
            'selected_xic_pair': [],
        }
        for pred_st, pred_ed in pred_times:
            auto_selected_xic = find_best_transition_index(time, xic, pred_st, pred_ed)
            ret[f'selected_xic_pair'].append(auto_selected_xic)

            for i, k in enumerate(['light', 'heavy']):
                summed_xic = xic[i, auto_selected_xic, :].sum(axis=0)
                peak_area, background = calculate_peak_area_with_time(
                                            time, summed_xic, pred_st, pred_ed)
                ret[f'{k}_area'].append(peak_area)
                ret[f'{k}_background'].append(background)
        
        pred_results[index] = ret
        
        if peptide_id_col is not None:
            pred_results[index][peptide_id_col] = sample[peptide_id_col]

    # restore original transform
    test_ds.set_transform(original_transforms)

    return pred_results

def run_deepmrm(model_path, input_ds):

    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f'there is no model file in {model_path}')

    if not isinstance(input_ds, DeepMrmInputDataset):
        raise ValueError('input_ds should be an instance of DeepMrmInputDataset')

    model = torch.load(model_path, map_location=device)
    transform = T.Compose([MakeInput(), TransitionDuplicate()])
    input_ds.set_transform(transform)    

    data_loader = torch.utils.data.DataLoader(
                            input_ds, 
                            batch_size=1, 
                            num_workers=0, 
                            collate_fn=obj_detection_collate_fn)
    output_df = model.evaluate(data_loader)

    result_dict = create_prediction_results(input_ds, output_df)
    # logger.info(f'Complete predictions for {ms_path.name}')

    return result_dict
