import torch

from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

from mstorch.data.manager import DataManager
from mstorch.tasks import ObjectDetectionTask
from mstorch.utils.logger import get_logger
from mstorch.utils.data.collate import SelectiveCollation
from mstorch.enums import PartitionType

from deepmrm import model_dir
from deepmrm.constant import XIC_KEY, TIME_KEY, TARGET_KEY
from deepmrm.data.dataset import DeepMrmDataset
from deepmrm.data_prep import get_metadata_df
from deepmrm.model.boundary_detect import BoundaryDetector
from deepmrm.transform.make_input import MakeTagets, MakeInput
from deepmrm.train.trainer import BaseTrainer
from deepmrm.transform.augment import RandomResizedCrop, TransitionJitter


logger = get_logger('DeepMRM')
num_workers = 4
gpu_index = 0
cycle_time = 0.5
RANDOM_SEED = 2022

task = ObjectDetectionTask('peak_detect', box_dim=1, num_classes=2)

# Define transforms
transform = T.Compose([
                MakeInput(force_resampling=True, use_rt=False, cycle_time=cycle_time),
                MakeTagets()
            ])

aug_transform = T.Compose([
        MakeInput(force_resampling=True, use_rt=False, cycle_time=cycle_time),
        TransitionJitter(p=0.25),
        RandomResizedCrop(p=0.7, cycle_time=cycle_time),
        MakeTagets()
    ])


def run_train(
    model_name, 
    augmentation, 
    returned_layers=[3, 4], 
    batch_size = 512, 
    num_epochs = 100,
    split_ratio=(.8, .1, .1),
    use_scl=False,
    backbone='resnet34',
    num_anchors=1):

    logger.info(f'Start loading dataset')
    label_df, pdac_xic, scl_xic = get_metadata_df(use_scl=use_scl)
    logger.info(f'Complete loading dataset')

    ds = DeepMrmDataset(
                label_df,
                pdac_xic,
                scl_xic,
                transform=transform)
    
    logger.info(f'The size of training-set: {len(ds)}')
    obj_detection_collate_fn = SelectiveCollation(exclusion_keys=[TARGET_KEY, TIME_KEY, XIC_KEY])

    data_mgr = DataManager(
                task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn, 
                split_ratio=split_ratio,
                random_seed=RANDOM_SEED)

    data_mgr.split()

    if augmentation:
        data_mgr.set_transform(aug_transform, partition_type=PartitionType.TRAIN)

    trainer = BaseTrainer(data_mgr, 
                        model_dir, 
                        logger, 
                        run_copy_to_device=False, 
                        gpu_index=gpu_index)

    model = BoundaryDetector(
                model_name, task, num_anchors=num_anchors, 
                returned_layers=returned_layers,
                backbone=backbone)
    
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = trainer.set_model(model).set_optimizer(optimizer).set_scheduler(scheduler)

    ## start training
    trainer.train(num_epochs=num_epochs, batch_size=batch_size)



if __name__ == "__main__":
    
    run_train("DeepMRM_BD", backbone='resnet18', augmentation=True)

    # run_train(
    #     "DeepMRM_Model_SCL",
    #     use_scl=True, 
    #     augmentation=True, 
    #     backbone='resnet34',
    #     num_epochs=200, 
    #     # split_ratio=(0.85, 0.15, 0.0),
    # )
