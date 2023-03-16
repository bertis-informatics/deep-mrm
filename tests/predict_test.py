import pandas as pd

from mstorch.data.mass_spec.tolerance import Tolerance, ToleranceUnit
from deepmrm.data.transition import TransitionData
from deepmrm.data.dataset.mrm import MRMDataset
from deepmrm.predict.input_xic import XicData, DeepMrmInputDataset
from deepmrm.predict.interface import run_deepmrm
from deepmrm import model_dir, project_dir


def test_interface():
    data_dir = project_dir / 'sample_data'
    # model_path = model_dir / 'DeepMRM_Model.pth'
    transition_path = data_dir / 'sample_target_list.csv'
    ms_path = data_dir / 'sample_mrm_data.mzML'

    peptide_id_col = 'peptide_id'
    mzml_path = ms_path
    all_trans_df = pd.read_csv(transition_path)
    mz_tol = Tolerance(0.5, ToleranceUnit.MZ)

    transition_data = TransitionData(all_trans_df, peptide_id_col=peptide_id_col)
    transform = None

    ds = MRMDataset(mzml_path, transition_data, transform=transform)
    ds.extract_data(tolerance=mz_tol)

    input_ds = DeepMrmInputDataset()
    for idx in range(len(ds)):
        sample = ds[idx]
        xic_data = XicData()
        for i in range(len(sample['XIC']['light'])):
            light_time = sample['XIC']['light'][i][0]
            light_intensity = sample['XIC']['light'][i][1]
            
            heavy_time = sample['XIC']['heavy'][i][0]
            heavy_intensity = sample['XIC']['heavy'][i][1]    

            xic_data.add_xic_pair(light_time, light_intensity, heavy_time, heavy_intensity)
        input_ds.add_xic_data(xic_data)

    pred_result = run_deepmrm(model_dir, input_ds)


    assert len(pred_result) == len(ds), "#outputs doesn't match to #inputs"
