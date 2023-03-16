from pathlib import Path
import joblib
import torch
import joblib

from mstorch.data.manager import DataManager
from mstorch.enums import PartitionType
from deepmrm import model_dir, private_project_dir
from deepmrm.data_prep import get_metadata_df
from deepmrm.data.dataset import DeepMrmDataset
from deepmrm.utils.eval import compute_peak_detection_performance
from deepmrm.train.train_boundary_detector import (
    RANDOM_SEED,
    obj_detection_collate_fn
)

reports_dir = private_project_dir / 'reports'
batch_size = 128
num_workers = 8
dataset_name = 'PDAC'
model_name = 'DeepMRM_QS'
model_path = Path(model_dir) / f'{model_name}.pth'
model = torch.load(model_path)

label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=False, only_quantifiable_peak=False)
ds = DeepMrmDataset(
            label_df, 
            pdac_chrom_df,
            scl_chrom_df,
            transform=model.transform)

data_mgr = DataManager(
                model.task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn,
                random_seed=RANDOM_SEED)
data_mgr.split()
testset_loader = data_mgr.get_dataloader(PartitionType.TEST, batch_size=batch_size)

output_df = model.evaluate(testset_loader)
save_path = Path(reports_dir/f'{dataset_name}_QS_output_df.pkl')
output_df.to_pickle(save_path)


# output_df, map_result = compute_peak_detection_performance(output_df)
# joblib.dump((output_df, map_result), save_path)


