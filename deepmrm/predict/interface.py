import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as T

from mstorch.utils import get_logger
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm.constant import TIME_KEY, XIC_KEY
from deepmrm.utils.peak import calculate_peak_area
from deepmrm.utils.eval import select_xic
from deepmrm.utils.transition_select import find_best_transition_index
from deepmrm.data import obj_detection_collate_fn
from deepmrm.predict.input_xic import XicData, DeepMrmInputDataset
from deepmrm.data.dataset import MRMDataset, PRMDataset
from deepmrm.model import load_models

logger = get_logger('DeepMRM')
#device = torch.device('cpu')

_boundary_detector = None
_quality_scorer = None

def create_prediction_results(test_ds, output_df, peptide_id_col=None, quality_th=0.5):

    pred_results = {}
    for idx, row in enumerate(output_df.iterrows()):
        index, row = row
        sample = test_ds[idx]
        time = sample[TIME_KEY]
        xic = sample[XIC_KEY]
        # pred_labels = row['labels']
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
        }
        
        #for pred_tm, pred_qt in zip(pred_times, pred_quality):
        for pred_box, pred_qt in zip(pred_boxes, pred_quality):
            pred_box = pred_box.astype(np.int32)
            # auto_selected_xic = find_best_transition_index(time, xic, pred_st, pred_ed)
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
        
        if peptide_id_col is not None:
            ret[peptide_id_col] = sample[peptide_id_col]

        pred_results[index] = ret

    return pred_results


def _load_models(model_dir, boundary_threshold=0.05, nms_thresh=0.3, device=torch.device('cpu')):
    global _boundary_detector, _quality_scorer
    model_dir = Path(model_dir)

    if (_boundary_detector is None) or (_quality_scorer is None):
        _boundary_detector, _quality_scorer = load_models(model_dir=model_dir, device=device)

    _boundary_detector.detector.score_thresh = boundary_threshold
    _boundary_detector.detector.nms_thresh = nms_thresh
    
    return _boundary_detector, _quality_scorer


def _run_deepmrm(boundary_detector, quality_scorer, input_ds):

    transform = T.Compose([MakeInput()])
    input_ds.set_transform(transform)    

    data_loader = torch.utils.data.DataLoader(
                            input_ds, 
                            batch_size=1, 
                            num_workers=0, 
                            collate_fn=obj_detection_collate_fn)
    # output_df = boundary_detector.evaluate(data_loader)
    output_df = boundary_detector.predict(data_loader)
    output_df = quality_scorer.predict(data_loader.dataset, output_df)

    result_dict = create_prediction_results(input_ds, output_df)
    # logger.info(f'Complete predictions for {ms_path.name}')

    return result_dict


def run_deepmrm(model_dir, input_ds):
    # if not isinstance(input_ds, DeepMrmInputDataset):
    #     raise ValueError('input_ds should be an instance of DeepMrmInputDataset')
    
    boundary_detector, quality_scorer = _load_models(model_dir)

    return _run_deepmrm(boundary_detector, quality_scorer, input_ds)




def run_deepmrm_with_mzml(model_dir, data_type, mzml_path, transition_data, tolerance):
    
    boundary_detector, quality_scorer = _load_models(model_dir)
    transform = T.Compose([MakeInput()])

    if data_type == 'MRM':
        ds = MRMDataset(mzml_path, transition_data, transform=transform)
    else:
        ds = PRMDataset(mzml_path, transition_data, metadata_df=None, transform=transform)

    n_chroms = ds.ms_reader.num_chromatograms
    n_spectra = ds.ms_reader.num_spectra

    logger.info(f'Mass-spec data contains {n_chroms} chromatograms and {n_spectra} spectra')
    if data_type == 'MRM' and n_chroms < 2:
        raise ValueError('data_type was set to MRM, but there is no chromatograms')

    logger.info(f'Start extracting chromatograms for targeted peptides')
    ds.extract_data(tolerance=tolerance)
    logger.info(f'Complete extracting chromatograms for targeted peptides')

    return _run_deepmrm(boundary_detector, quality_scorer, ds)

    # data_loader = torch.utils.data.DataLoader(
    #                     ds, 
    #                     batch_size=1, 
    #                     num_workers=0, 
    #                     collate_fn=obj_detection_collate_fn)
    
    # logger.info(f'Start predictions')
    # output_df = model.evaluate(data_loader)
    # result_dict = create_prediction_results(ds, output_df, transition_data.peptide_id_col)
    # logger.info(f'Complete predictions')

    # result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    # return result_df