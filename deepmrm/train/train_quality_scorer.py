import torch

from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAccuracy

from mstorch.data.manager import DataManager
from mstorch.tasks import ClassificationTask
from mstorch.utils.logger import get_logger
from mstorch.enums import PartitionType

from deepmrm import model_dir
from deepmrm.data.dataset import DeepMrmDataset
from deepmrm.data_prep import get_metadata_df
from deepmrm.model.quality_score import QualityScorer
from deepmrm.transform.make_input import MakeInput
from deepmrm.transform.make_target import MakePeakQualityTarget
from deepmrm.train.trainer import BaseTrainer
from deepmrm.transform.augment import (
    RandomResizedCrop, MultiplicativeJitter,
    RandomRTShift, ReplicateTransitionWithNoise
)
from deepmrm.data import obj_detection_collate_fn

logger = get_logger('DeepMRM')
num_workers = 4
gpu_index = 0
RANDOM_SEED = 2022

# task = ObjectDetectionTask('peak_detect', box_dim=1, num_classes=2)
task = ClassificationTask('peak_quality', 
                           label_column='manual_quality',
                           prediction_column='predicted_quality',
                           num_classes=2)

# Define transforms
transform = T.Compose([
                MakeInput(force_resampling=True, use_rt=False),
                MakePeakQualityTarget()
            ])

aug_transform = T.Compose([
        MakeInput(force_resampling=True, use_rt=False),
        T.RandomChoice([
            ReplicateTransitionWithNoise(p=0.2),
            RandomRTShift(p=0.25), 
            MultiplicativeJitter(p=0.25, noise_scale_ub=1e-3, noise_scale_lb=1e-4),
        ]),
        RandomResizedCrop(p=0.7, manual_bd_only=False),
        MakePeakQualityTarget()
    ])


def run_qs_model(model_name, num_epochs=100, batch_size=512, augmentation=True):

    logger.info(f'Start loading dataset')
    label_df, pdac_xic, scl_xic = get_metadata_df(
                                        only_peak_boundary=False, 
                                        use_scl=False)
    logger.info(f'Complete loading dataset')

    ds = DeepMrmDataset(
                label_df, 
                pdac_xic,
                scl_xic,
                transform=transform)

    logger.info(f'The size of training-set: {len(ds)}')
    data_mgr = DataManager(
                task, 
                ds, 
                num_workers=num_workers, 
                collate_fn=obj_detection_collate_fn, 
                split_ratio=(.8, .1, .1),
                random_seed=RANDOM_SEED)
    data_mgr.split()

    if augmentation:
        data_mgr.set_transform(aug_transform, partition_type=PartitionType.TRAIN)

    trainer = BaseTrainer(data_mgr, 
                        model_dir, 
                        logger, 
                        run_copy_to_device=False, 
                        gpu_index=gpu_index)

    trainer.add_metrics(task, BinaryAccuracy())
    
    model = QualityScorer(model_name, task)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = trainer.set_model(model).set_optimizer(optimizer).set_scheduler(scheduler)

    ## 6. start training
    trainer.train(num_epochs=num_epochs, batch_size=batch_size)

# testset_loader = data_mgr.get_dataloader('test')
# ret_df = model.evaluate(testset_loader)
# ret_df['pred_quality'] = (ret_df['predicted_quality'] > 0.5).astype(int)

# m = (ret_df['manual_quality'] == 0) 
# m &= (ret_df['pred_quality'] == 1)

# from sklearn.metrics import accuracy_score
# from mstorch.evaluation.classification import ClassificationReport
# import numpy as np
# from matplotlib import pyplot as plt
# accuracy_score(ret_df['manual_quality'], ret_df['pred_quality'])

# y_prob = np.zeros((ret_df.shape[0], 2))
# y_prob[:, 1] = ret_df['predicted_quality'].values
# y_prob[:, 0] = 1 -y_prob[:, 1]

# rpt = ClassificationReport(2, ['Poor', 'Quantifiable'])
# rpt.add_experiment_result( 
#     1, ret_df['manual_quality'].values, y_prob)

# conf_disp = rpt.get_confusion_matrix_display(1)

# conf_disp.plot()
# plt.savefig('./temp/conf_matrix.jpg')

# plt.figure()
# fig, ax = rpt.get_roc_curve(1)
# plt.savefig('./temp/roc_curve.jpg')

if __name__ == "__main__":
    from deepmrm.train.train_boundary_detector import run_train
    
    run_train(
        "DeepMRM_BD_NoAug",
        aug_transform=None,
        backbone='resnet18', 
        batch_size = 512,
        num_epochs = 100,
        only_peak_boundary=False,
    )

    run_qs_model(
        'DeepMRM_QS_NoAug', 
        num_epochs=100, 
        augmentation=False
    )


    