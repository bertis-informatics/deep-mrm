import argparse
import pandas as pd
from pathlib import Path
import torch

from mstorch.utils import get_logger
from deepmrm import model_dir
from deepmrm.data.transition import TransitionData
from deepmrm.predict.interface import run_deepmrm_with_mzml
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit


# model_file_path = model_dir / 'DeepMRM_Model.pth'
parser = argparse.ArgumentParser()

parser.add_argument("-model_dir", type=str, help="directory for model files", default=model_dir)
parser.add_argument("-data_type", type=str, help="data type (MRM, PRM or DIA)", default='MRM')
parser.add_argument("-target", type=str, help="target tsv path")
parser.add_argument("-input", type=str, help="mass spec file path")
# parser.add_argument("-peptide_id_col", type=str, help="peptide ID column", default='sequence')
parser.add_argument("-tolerance", type=float, help="Ion match tolerance (in PPM)", default=20)

args = parser.parse_args()
logger = get_logger('DeepMRM')
device = torch.device('cpu')



if __name__ == "__main__":

    if (args.input is None) or (args.target is None):
        raise ValueError('mass-spec file (MzML) and target csv file should be specified')

    model_path = Path(args.model)
    ms_path = Path(args.input)
    transition_path = Path(args.target)
    #peptide_id_col = args.peptide_id_col

    peptide_id_col = 'peptide_id'
    tol = args.tolerance
    data_type = args.data_type
    output_path = ms_path.parent

    if model_path.exists():
        logger.info(f'Load model file from: {model_path.absolute()}')
        model = torch.load(model_path, map_location=device)
    else:
        raise ValueError(f'Cannot file model file in {model_path.absolute()}')

    if transition_path.exists():
        logger.info(f'Load transition file from: {transition_path.absolute()}')
        all_trans_df = pd.read_csv(transition_path)
        
        for col in ['peptide_id', 'precursor_mz', 'product_mz', 'is_heavy']:
            if col not in all_trans_df.columns:
                raise ValueError(f'Cannot find {col} column in the target list file')
    else:
        raise ValueError(f'Cannot find transition csv file in {transition_path.absolute()}')

    if ms_path.exists():
        logger.info(f'Load mass-spec file from: {ms_path.absolute()}')
        if ms_path.suffix.lower() != '.mzml':
            raise ValueError('Mass-spec file should be mzML format')
    else:
        raise ValueError(f'Cannot find mass-sepc file in {ms_path.absolute()}')

    if data_type not in ['PRM', 'MRM', 'DIA']:
        raise ValueError('data_type should be one of PRM, MRM, or DIA')
    
    logger.info(f'Arguments:\n{args}')

    transition_data = TransitionData(all_trans_df, peptide_id_col=peptide_id_col)
    tolerance = Tolerance(tol, ToleranceUnit.MZ)

    result_dict = run_deepmrm_with_mzml(
                    model, 
                    data_type, 
                    ms_path, 
                    transition_data, 
                    tolerance)
    
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    
    save_path = output_path/ f'{ms_path.stem}_DeepMRM.csv'
    result_df.to_csv(save_path, index=False)
    logger.info(f'Save result csv file: {save_path}')