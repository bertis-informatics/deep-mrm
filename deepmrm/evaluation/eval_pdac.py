from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch

from mstorch.data.manager import DataManager
from mstorch.enums import PartitionType
from mstorch.utils.data.collate import SelectiveCollation

from deepmrm import model_dir
from deepmrm.constant import RT_KEY, XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.data_prep import get_metadata_df, pdac, scl
from deepmrm.data.dataset import DeepMrmDataset
from torchvision import transforms as T
from deepmrm.transform.make_input import MakeInput
from deepmrm.train.run import RANDOM_SEED

#from torchvision.models.detection.transform import GeneralizedRCNNTransform
from deepmrm.train.run import task, transform
from deepmrm.utils.eval import compute_peak_detection_performance
import joblib

batch_size = 128
num_workers = 8

model_names = [
    # 'ResNet18_Aug', 
    'ResNet34_Aug_1x', 
    # 'ResNet50_Aug', 
    # 'ResNet34_NoAug', 
    # 'ResNet34_NoAug_NoGrpConv',
    # 'ResNet34_Aug_NoGrpConv',
]

label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=False, only_quantifiable_peak=True)


ds = DeepMrmDataset(
            label_df, 
            pdac_chrom_df,
            scl_chrom_df,
            transform=transform)

obj_detection_collate_fn = SelectiveCollation(exclusion_keys=[TARGET_KEY, TIME_KEY, XIC_KEY])
data_mgr = DataManager(
                task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn,
                random_seed=RANDOM_SEED)
data_mgr.split()

testset_loader = data_mgr.get_dataloader(PartitionType.TEST, batch_size=batch_size)
test_ds = testset_loader.dataset

#test_ds.metadata_df = test_ds.metadata_df.sample(10)
dataset_name = 'PDAC'
iou_thresholds = [0.3]
max_detection_thresholds = [1, 3]


# for model_name in model_names:
model_name = model_names[0]
model_path = Path(model_dir) / f'{model_name}.pth'
model = torch.load(model_path)
transform = model.transform
task = model.task
# model.set_detections_per_img(20)

output_df = model.evaluate(testset_loader)

output_df, map_result = compute_peak_detection_performance(
                                    test_ds, 
                                    output_df,
                                    iou_thresholds=iou_thresholds, 
                                    max_detection_thresholds=max_detection_thresholds)

save_path = Path(f'reports/{dataset_name}_eval_{model_name}.pkl')
    

joblib.dump((output_df, map_result), save_path)




s = label_df['manual_frag_quality_t1'] + label_df['manual_frag_quality_t2'] + label_df['manual_frag_quality_t3']

m = s == 0
label_df[m]
label_df.loc[138, :]


# how to score the quantifiability of peak group
 - peak shape
 - h/r ratio 
 