import torch
from pathlib import Path

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
from deepmrm.model.model import DeepMrmModel
from deepmrm.transform.make_input import MakeTagets, MakeInput
from deepmrm.train.trainer import BaseTrainer
from deepmrm.transform.augment import (
    TransitionShuffle, 
    RandomResizedCrop,
    TransitionJitter
)

logger = get_logger('DeepMRM')
num_workers = 4
gpu_index = 0



def run_train(
    model_name, 
    augmentation, 
    returned_layers=[3, 4], 
    pre_trained=False, 
    batch_size = 512, 
    num_epochs = 100,
    change_conv1=True,
    split_ratio=(.8, .1, .1),
    use_scl=False,
    backbone='resnet34',
    num_anchors=1,
    cycle_time=0.5
    ):

    label_df, pdac_chrom_df, scl_chrom_df = get_metadata_df(use_scl=use_scl)

    # Define transforms
    transform = T.Compose([
                    MakeInput(force_resampling=True, use_rt=False, cycle_time=cycle_time),
                    MakeTagets()
                ])

    aug_transform = T.Compose([
            MakeInput(force_resampling=True, use_rt=False, cycle_time=cycle_time),
            # T.RandomChoice([
            #     TransitionRankShuffle(p=0.05),
            #     RandomRTShift(p=0.05),
            # ]),
            TransitionShuffle(p=0.5),
            TransitionJitter(p=0.25),
            RandomResizedCrop(p=0.7, cycle_time=cycle_time),
            MakeTagets()
        ])


    ds = DeepMrmDataset(
                label_df, 
                pdac_chrom_df,
                scl_chrom_df,
                transform=transform)

    obj_detection_collate_fn = SelectiveCollation(exclusion_keys=[TARGET_KEY, TIME_KEY, XIC_KEY])

    num_classes = label_df['manual_quality'].nunique()
    task = ObjectDetectionTask('peak_detect', box_dim=1, num_classes=num_classes)

    data_mgr = DataManager(
                task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn, 
                split_ratio=split_ratio,
                random_seed=2022)

    data_mgr.split()

    if augmentation:
        data_mgr.set_transform(aug_transform, partition_type=PartitionType.TRAIN)

    trainer = BaseTrainer(data_mgr, 
                        model_dir, 
                        logger, 
                        run_copy_to_device=False, 
                        gpu_index=gpu_index)
    
    if pre_trained:
        model_path = Path(model_dir) / f'{model_name}.pth'
        model = torch.load(model_path)
        model.name = f'{model_name}_refined'
    else:
        model = DeepMrmModel(
                    model_name, task, num_anchors=num_anchors, 
                    returned_layers=returned_layers,
                    backbone=backbone,
                    change_conv1=change_conv1)
    
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = trainer.set_model(model).set_optimizer(optimizer).set_scheduler(scheduler)

    ## 6. start training
    trainer.train(num_epochs=num_epochs, batch_size=batch_size)

if __name__ == "__main__":
    
    run_train(
        "DeepMRM_Model", 
        use_scl=True, cycle_time=0.5,
        augmentation=True, pre_trained=False,
        returned_layers=[3, 4], num_anchors=1, backbone='resnet34',
        num_epochs=200, split_ratio=(0.85, 0.15, 0.0),
    )    

    # run_train(
    #     "ResNet34_Aug", 
    #     augmentation=True, pre_trained=False,
    #     returned_layers=[3, 4], num_anchors=1, backbone='resnet34',
    #     num_epochs=200, split_ratio=(0.85, 0.15, 0.0),
    # )    

    # run_train(
    #     "ResNet18_Aug", 
    #     augmentation=True, pre_trained=False,  
    #     returned_layers=[3, 4], num_anchors=1, backbone='resnet18'
    # )
    
    # run_train(
    #     "ResNet50_Aug", 
    #     augmentation=True, pre_trained=False,  
    #     returned_layers=[3, 4], num_anchors=1, backbone='resnet50',
    # )        
    
    # run_train(
    #     "ResNet34_Aug_NoGrpConv", 
    #     augmentation=True, pre_trained=False,  
    #     returned_layers=[3, 4], num_anchors=1, change_conv1=False,
    # )
    
    # run_train(
    #     "ResNet34_NoAug_NoGrpConv", 
    #     augmentation=False, pre_trained=False,  
    #     returned_layers=[3, 4], num_anchors=1, change_conv1=False,
    # )    

    # run_train(
    #     "ResNet34_NoAug", 
    #     augmentation=False, pre_trained=False,  
    #     returned_layers=[3, 4], num_anchors=1
    # )

