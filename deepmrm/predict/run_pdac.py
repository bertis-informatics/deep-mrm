from time import time as timer
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as T

from mstorch.utils import get_logger

from deepmrm.data import obj_detection_collate_fn
from deepmrm.transform.make_input import MakeInput, TransitionDuplicate
from deepmrm import model_dir
from deepmrm.data.transition import TransitionData
from deepmrm.data.dataset import MRMDataset
from deepmrm.predict.interface import create_prediction_results
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm import get_yaml_config, model_dir
from deepmrm.data_prep import pdac

model_path = model_dir / 'ResNet34_Aug.pth'
device = torch.device('cpu')

logger = get_logger('DeepMRM')

_conf = get_yaml_config()
conf = _conf['PDAC-SIT']

ROOT_DIR = Path(conf['ROOT_DIR'])
MZML_DIR = Path(conf['MZML_DIR'])
RESULT_DIR = ROOT_DIR / 'DeepMRM_results'
LABEL_DIR = MZML_DIR.parent / 'labels'

def batch_run_pdac_dataset():
    t_started = timer()    

    label_df = pdac.get_label_df()
    mzml_files = label_df['mzML'].unique()
    mz_tol = Tolerance(0.5, ToleranceUnit.MZ)

    model = torch.load(model_path, map_location=device)

    for mzml_fname in mzml_files:

        if mzml_fname == 'PDAC_7_1_Nano_ESI_dMRM_MFC_5LPM_2500V_225C_5per_Sol_B_02per_TFA_C2_051021.mzML': 
            break

        mzml_path = MZML_DIR / mzml_fname
        label_path = (LABEL_DIR / mzml_fname).with_suffix('.tsv')

        trans_df = pd.read_csv(label_path, sep='\t')

        transition_data = TransitionData(
                            trans_df, 
                            peptide_id_col='peptide_code',
                            is_heavy_col='is_heavy',
                            precursor_mz_col='precursor_mz',
                            product_mz_col='product_mz')

        transform = T.Compose([MakeInput(), TransitionDuplicate()])

        ds = MRMDataset(mzml_path, transition_data, transform=transform)
        ds.extract_data(tolerance=mz_tol)

        data_loader = torch.utils.data.DataLoader(
                                ds, 
                                batch_size=64, 
                                num_workers=4, 
                                collate_fn=obj_detection_collate_fn)
        output_df = model.evaluate(data_loader)
        result_dict = create_prediction_results(ds, output_df, transition_data.peptide_id_col)
        result_df = pd.DataFrame.from_dict(result_dict, orient='index')

        result_df.to_csv('temp.csv')
        logger.info(f'Complete predictions for {mzml_fname}')

    t_ended = timer()
    logger.info(f'Total running time for {len(mzml_files)} files: {t_ended-t_started}')

    # save_path = output_path/ f'{ms_path.stem}_predictions.csv'
    # pd.DataFrame.from_dict(result_dict, orient='index').to_csv(save_path, index=True)
    # logger.info(f'Saved result csv file: {save_path}')



if __name__ == "__main__":
    batch_run_pdac_dataset()