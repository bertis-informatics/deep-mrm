from pathlib import Path
import torch
from mstorch.utils.logger import get_logger

logger = get_logger('DeepMRM')

def load_models(model_dir, device):

    model_dir = Path(model_dir)
    
    model_path = model_dir/'DeepMRM_BD.pth'
    if model_path.exists():
        logger.info(f'load model from {model_path}')
        boundary_detector = torch.load(model_path, map_location=device)
        # boundary_detector.detector.score_thresh = 0.0
    else:
        raise ValueError(f'Cannot find {model_path}')    

    model_path = model_dir/'DeepMRM_QS.pth'
    if model_path.exists():
        logger.info(f'load model from {model_path}')
        quality_scorer = torch.load(model_path, map_location=device)
    else:
        raise ValueError(f'Cannot find {model_path}')    

    return boundary_detector, quality_scorer
