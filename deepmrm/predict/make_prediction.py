import argparse
import pandas as pd
from pathlib import Path
import torch

from mstorch.utils import get_logger
from deepmrm import model_dir
from deepmrm.data.transition import TransitionData
from deepmrm.predict.interface import run_deepmrm_with_mzml, get_top1_result_df
from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit


parser = argparse.ArgumentParser()
# parser.add_argument("-model_dir", type=str, help="directory for model files", default=model_dir)
# parser.add_argument("-peptide_id_col", type=str, help="peptide ID column", default='sequence')
# parser.add_argument("-data_type", type=str, help="data type (MRM, PRM or DIA)", default='MRM')
parser.add_argument("-target", type=str, help="target csv path")
parser.add_argument("-input", type=str, help="mass spec file path")
parser.add_argument("-tolerance", type=float, help="mass match tolerance (in PPM)", default=10)
args = parser.parse_args()
logger = get_logger('DeepMRM')


if __name__ == "__main__":

    if (args.input is None) or (args.target is None):
        raise ValueError('mass-spec file (MzML) and target csv file should be specified')

    ms_path = Path(args.input)
    transition_path = Path(args.target)
    tol = args.tolerance
    peptide_id_col = 'peptide_id'
    output_path = ms_path.parent

    logger.info(f'Arguments: {args}')

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

    transition_data = TransitionData(all_trans_df, peptide_id_col=peptide_id_col)
    logger.info(f'Load total {transition_data.num_targets} target peptides')
    tolerance = Tolerance(tol, ToleranceUnit.PPM)

    raw_result_df = run_deepmrm_with_mzml(
                    model_dir, 
                    ms_path, 
                    transition_data, 
                    tolerance)
    
    result_df = get_top1_result_df(peptide_id_col, raw_result_df)
    
    save_path = output_path/ f'{ms_path.stem}_DeepMRM_top1.csv'
    result_df.to_csv(save_path, index=False)

    save_path = output_path/ f'{ms_path.stem}_DeepMRM.csv'
    raw_result_df.to_csv(save_path, index=False)
    logger.info(f'Save result csv file: {save_path}')