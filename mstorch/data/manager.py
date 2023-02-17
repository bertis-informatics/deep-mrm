import copy
import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.utils import check_random_state

from mstorch.data.dataset import BaseDataset
from mstorch.enums import PartitionType, SplitMethod
from mstorch.tasks import BaseTask
from mstorch.utils import get_logger

DEFAULT_NUM_WORKERS = 8
DEFAULT_SPLIT_RATIO = (.8, .1, .1) # train, valid, test

logger = get_logger()


class DataManager(object):
    """
    An utility class for handling DataSet instance.
    It splits and constructs datasets, and returns dataloaders encapulating transformers.
    """

    def __init__(self,
                 task,
                 dataset,
                 split_ratio=DEFAULT_SPLIT_RATIO, 
                 num_workers=DEFAULT_NUM_WORKERS,
                 random_seed=1234,
                 collate_fn=None):

        assert type(random_seed) == int
        
        if not isinstance(task, BaseTask):
            raise ValueError(
                "'task' must be an instance of BaseTask")

        if not isinstance(dataset, BaseDataset):
            raise ValueError(
                "'dataset' must be an instance of BaseDataset")

        assert len(split_ratio) == 3

        self.task = task
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.random_state = check_random_state(random_seed)
        self.datasets = None
        self.splitted = False
        self.kfold_index_set = None
        self.patient_id_column = None
        self.stratify_column = None
        self.samplers = dict()
        self.collate_fn = collate_fn


    @property
    def num_samples(self):
        """
        Returns:
            int: total number of samples
        """
        if self.splitted:
            return sum(ds.metadata_df.shape[0] for ds in self.datasets.values())
        return self.metadata_df.shape[0]

    @property
    def num_classes(self):
        if self.task.is_classification():
            return self.task.num_classes
        else:
            return 0

    def _split_kfold(self):

        n_splits = int(np.sum(self.split_ratio) / self.split_ratio[2])
        meta_df = self.dataset.metadata_df
        y, groups = None, None
        
        if self.patient_id_column is not None:
            kf = GroupKFold(n_splits=n_splits)
            groups = meta_df[self.patient_id_column]
        elif self.stratify_column is not None:
            kf = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)
            y = meta_df[self.stratify_column]
        else:
            kf = KFold(n_splits=n_splits, random_state=self.random_state)

        fold_id = 0
        fold_index_set = dict()
        for train_index, test_index in kf.split(meta_df, y=y, groups=groups):
            fold_index_set[fold_id] = {
                'train_index':train_index,
                'test_index': test_index,
            }
            fold_id += 1
        self.kfold_index_set = fold_index_set
        assert len(self.kfold_index_set) == n_splits
        logger.info(f"dataset has been splitted consecutive {n_splits} folds")

    def take_kfold(self, fold_index):

        assert (self.splitted and self.kfold_index_set)
        assert fold_index < len(self.kfold_index_set)

        train_index = self.kfold_index_set[fold_index]['train_index']
        test_index = self.kfold_index_set[fold_index]['test_index']
        train_ds_, test_ds = self.dataset.split_by_index(train_index, test_index)

        # split train_ds_ into train and val datasets
        val_size = self.split_ratio[1]/np.sum(self.split_ratio[:2])
        train_ds, val_ds = train_ds_.split(
                                val_size, 
                                patient_id_column=self.patient_id_column, 
                                stratify_column=self.stratify_column,
                                random_state=self.random_seed)

        self.datasets = {
                PartitionType.TRAIN: train_ds,
                PartitionType.VALIDATION: val_ds,
                PartitionType.TEST: test_ds,
            }
        
        logger.info(f"Fold-{fold_index} has been taken")


    def _split_holdout(self, dataset, patient_id_column, stratify_column):

        test_size = self.split_ratio[2]/np.sum(self.split_ratio)      
        if test_size > 0:
            train_ds_, test_ds = dataset.split(
                                    test_size, 
                                    patient_id_column=patient_id_column, 
                                    stratify_column=stratify_column,
                                    random_state=self.random_state)
        else:
            train_ds_ = dataset
            empty_meta_df = pd.DataFrame(columns=dataset.metadata_df.columns)
            test_ds = copy.copy(dataset).set_metadata(empty_meta_df)

        val_size = self.split_ratio[1]/np.sum(self.split_ratio[:2])
        train_ds, val_ds = train_ds_.split(
                                val_size, 
                                patient_id_column=patient_id_column, 
                                stratify_column=stratify_column,
                                random_state=self.random_state)
        return train_ds, val_ds, test_ds


    def split(self, patient_id_column=None, stratify_column=None, method=SplitMethod.HOLDOUT):
        """
        Split entire dataset into 3 datasets: train, valid, and test
        If patient_id_column is provided, it splits data according to patient id.
        """
        
        method = SplitMethod(method)

        self.patient_id_column = patient_id_column
        self.stratify_column = stratify_column
        self.splitted = True

        if method == SplitMethod.KFOLD:
            self._split_kfold()
            return
        
        train_ds, val_ds, test_ds = None, None, None
        if method == SplitMethod.HOLDOUT:
            train_ds, val_ds, test_ds = self._split_holdout(
                            self.dataset, patient_id_column, stratify_column)
        elif method == SplitMethod.HOLDOUT_PER_CLASS:
            if not self.task.is_classification():
                raise ValueError(f'{method} only works for classification task')

            if self.task.is_multilabel():
                raise ValueError(f'{method} only works for single label task')

            for class_ in self.task.get_classes():
                label = self.task.labels[0]
                mask = self.dataset.metadata_df[label] == class_
                df_ = self.dataset.metadata_df[mask]
                ds_ = copy.copy(self.dataset).set_metadata(df_)
                ds_tuple = self._split_holdout(ds_, patient_id_column, stratify_column)
                
                if train_ds is None:
                    train_ds, val_ds, test_ds = ds_tuple
                else:
                    train_ds.append(ds_tuple[0])
                    val_ds.append(ds_tuple[1])
                    test_ds.append(ds_tuple[2])
                    
            train_ds.metadata_df = train_ds.metadata_df.reset_index(drop=True)
            val_ds.metadata_df = val_ds.metadata_df.reset_index(drop=True)
            test_ds.metadata_df = test_ds.metadata_df.reset_index(drop=True)
        else:
            raise NotImplementedError()

        self.datasets = {
                PartitionType.TRAIN: train_ds,
                PartitionType.VALIDATION: val_ds,
                PartitionType.TEST: test_ds,
            }
        
        if self.task.is_classification():
            split_stat_df = self.get_split_stats()
            logger.info("dataset has been splitted\n{}".format(split_stat_df))

    
    def get_split_stats(self):

        if not self.task.is_classification() and not self.task.is_object_detection():
            raise NotImplementedError('Only works for classification task and object detection task')

        patient_id_column = self.patient_id_column

        stat_df = None
        temp_datasets = {'whole': self.dataset}
        temp_datasets.update(
            {k.value: v for k, v in self.datasets.items()})

        basic_stat = {'total': dict()}
        for partition_type, ds in temp_datasets.items():
            n_patients = ds.metadata_df[patient_id_column].nunique() if patient_id_column is not None else 0
            basic_stat['total'][(partition_type, '#samples')] = len(ds)
            basic_stat['total'][(partition_type, '#patients')] = n_patients

        basic_stat_df = pd.DataFrame.from_dict(basic_stat, orient='index')            

        if self.task.is_classification():
            label_col = self.task.label_column
            for partition_type, ds in temp_datasets.items():
                sample_cnt = ds.metadata_df.groupby(label_col).count().iloc[:, 0]\
                            .rename((partition_type, '#samples'))\
                            .to_frame()
                if patient_id_column is not None:
                    patient_cnt = ds.metadata_df.groupby(label_col)[patient_id_column].nunique()\
                                    .rename((partition_type, '#patients'))
                    sample_cnt = sample_cnt.join(patient_cnt)
                stat_df = sample_cnt if stat_df is None else stat_df.join(sample_cnt)
            basic_stat_df = basic_stat_df.append(stat_df)
        basic_stat_df = basic_stat_df.fillna(0)

        return basic_stat_df


    def set_sampler(self, sampler, partition_type):
        """[summary]

        Args:
            sampler (torch.utils.data.Sampler): dataset에서 샘플링할 때 사용하게 될 sampler 인스턴스
            partition_type (PartitionType): sampler를 적용할 dataset

        Returns:
            DataManager:
        """
        
        assert isinstance(sampler, torch.utils.data.Sampler)
        partition_type = PartitionType(partition_type)
        self.samplers[partition_type] = sampler
        
        return self

    def set_transform(self, transform, partition_type=None):
        """reset transform for the specified dataset 

        Args:
            transform (Transform): transform instance
            partition_type (PartitionType): transform을 적용할 dataset
        """

        if partition_type is None:
            self.dataset.set_transform(transform)
            if self.datasets:
                for k, ds in self.datasets.items():
                    self.datasets[k] = ds.set_transform(transform)
        else:
            partition_type = PartitionType(partition_type)
            self.datasets[partition_type].set_transform(transform)

        return self

    def get_dataset(self, partition_type):
        """
        Args:
            partition_type (PartitionType): partition to be selected

        Returns:
            [DataSet]: dataset instance corresponding to the selected partition
        """

        assert self.splitted, "dataset must be splitted"
        partition_type = PartitionType(partition_type)
        return self.datasets[partition_type]

    def get_dataloader(self, partition_type, batch_size=32):

        partition_type = PartitionType(partition_type)

        ds = self.get_dataset(partition_type)
        sampler = self.samplers[partition_type] if partition_type in self.samplers else None
        
        # create data loader instance
        data_loader = torch.utils.data.DataLoader(
                        ds,
                        batch_size=batch_size, 
                        shuffle=(partition_type==PartitionType.TRAIN and sampler is None),
                        num_workers=self.num_workers,
                        sampler=sampler,
                        collate_fn=self.collate_fn)
        
        return data_loader
