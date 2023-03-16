from pathlib import Path
import torch
import joblib
import pandas as pd

from mstorch.data.manager import DataManager
from mstorch.enums import PartitionType
from deepmrm import model_dir, private_project_dir
from deepmrm.data_prep import get_metadata_df
from deepmrm.data.dataset import DeepMrmDataset
from deepmrm.utils.eval import compute_peak_detection_performance
from deepmrm.data import obj_detection_collate_fn
from deepmrm.train.train_boundary_detector import RANDOM_SEED


reports_dir = private_project_dir / 'reports'
batch_size = 128
num_workers = 8
dataset_name = 'PDAC'

model_name = 'DeepMRM_BD'
model_path = Path(model_dir) / f'{model_name}.pth'
model = torch.load(model_path)

model_name = 'DeepMRM_QS'
model_path = Path(model_dir) / f'{model_name}.pth'
model_qs = torch.load(model_path)


label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=False, only_quantifiable_peak=True)

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

output_df = model.predict(testset_loader)

output_df = model_qs.predict(testset_loader.dataset, output_df)

save_path = Path(reports_dir/f'{dataset_name}_output_df.pkl')
output_df.to_pickle(save_path)

# test_ds = data_mgr.get_dataset('test')
# output_df = pd.read_pickle(save_path)

# map_result, output_df = compute_peak_detection_performance(output_df)
# joblib.dump((output_df, map_result), save_path)


