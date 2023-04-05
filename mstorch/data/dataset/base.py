from abc import ABC, abstractmethod
import copy
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

import torch.utils.data
import torch.nn
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import (
    ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit
)
from pyopenms import OnDiscMSExperiment
from mstorch.utils import get_logger

logger = get_logger()

class BaseDataset(torch.utils.data.Dataset, ABC):

    def __init__(self, 
                 metadata_df,
                 transform=None):
        self.set_transform(transform)
        self.set_metadata(metadata_df)
    

    def __getitem__(self, index):
        """ 데이터 index에 해당하는 데이터를 읽어와서 리턴
        metadata의 특정 row를 dictionary 형태로 변환하고, 해당 데이터와 연관된 영상 데이터를 읽어와서 
        dictionary에 추가하여 리턴

        Args:
            index ([int]): metadata_df의 row index

        Raises:
            NotImplementedError: [description]
        """
        # For available return-types, please refer to:
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        # Typically, we will use dictionary data structure.
        if torch.is_tensor(index):
            index = index.tolist()        
        
        sample = self.metadata_df.iloc[index, :]
        sample_index = sample.name
        
        sample = sample.to_dict()
        sample[self.metadata_index_name] = sample_index

        return sample
    
    def get_hash_string(self):
        """현재 dataset에 대한 hash 값을 string 형태로 생성

        Returns:
            [str]: hash string for dataset instance
        """
        hash_vals = pd.util.hash_pandas_object(self.metadata_df)
        hash_obj = hashlib.sha256(np.concatenate((hash_vals.values, [hash(self.transform)])))
        return hash_obj.hexdigest()

    def __hash__(self):
        return hash(self.get_hash_string())

    def __len__(self):
        return len(self.metadata_df)

    def load_image(self, img_path):
        return default_loader(img_path)

    def open_msdata(self, ms_fpath):
        od_exp = OnDiscMSExperiment()
        od_exp.openFile(str(ms_fpath))
        return od_exp        

    def set_metadata(self, metadata_df):
        
        if np.any(metadata_df.index.duplicated()):
            raise ValueError('There are duplicated indexes in metadata_df')

        metadata_df = metadata_df.copy()
        
        # # pytorch DataLoader에서 data collate할 때, tensor로 변환하게 되는데 datetime 형식 등의 데이터들은 별도의 처리가 필요함.
        # # custom collate_fn을 구현할 수도 있겠으나, simplicity를 위해 이들을 string이나 integer로 변환.
        # # 참고: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
        # exclude_columns = list()
        # for col in metadata_df.select_dtypes(include=['bool', 'datetime', np.object_]):
        #     if metadata_df[col].dtype == 'bool':
        #         metadata_df[col] = metadata_df[col].astype(np.int64)
        #     elif metadata_df[col].dtype.type == np.datetime64:
        #         metadata_df[col] = metadata_df[col].astype(str)
        #     else:
        #         try:
        #             metadata_df[col] = metadata_df[col].astype(str)
        #         except:
        #             logger.warning(f'{col} type conversion fail')
        #             exclude_columns = [col]
        
        # if len(exclude_columns) > 0:
        #     # 변환에 실패한 컬럼은 아예 metadata_df 에서 제외
        #     logger.warning(f'{exclude_columns} columns are excluded from metadata')
        #     metadata_df = metadata_df.select_dtypes(exclude=exclude_columns)

        self.metadata_df = metadata_df
        return self

    @property
    def metadata_index_name(self):
        if self.metadata_df.index.name:
            return self.metadata_df.index.name
        return 'index'

    def set_transform(self, transform):
        assert transform is None or isinstance(transform, T.Compose)
        self.transform = transform
        return self

    def split(self, test_size, patient_id_column=None, stratify_column=None, random_state=None):
        """데이터셋을 2개의 dataset으로 split.

        Args:
            test_size ([float or int]): float인 경우 0-1.0 사이의 비율 값을 나타내며, 
                                        int인 경우 샘플의 절대 사이즈를 의미한다.
            patient_id_column ([str], optional): patient_id_column이 존재하는 경우 
                                        해당 patient_id_column 을 기준으로 split을 하게 된다. 
                                        Defaults to None.
            stratify_column ([str], optional): stratify_column이 specified 된 경우 (예, 레이블 컬럼)
                                        해당 컬럼을 기준으로 stratified split하게 된다. Defaults to None.
            random_state ([type], optional): reproducibility를 위해서 random_state를 specify 할 수 있다. 
                                        Defaults to None.

        Returns:
            [Tuple of Dataset]: 분리된 2개의 Dataset
        """
        
        df = self.metadata_df

        if patient_id_column is not None:
            gss = GroupShuffleSplit(n_splits=1, 
                                    test_size=test_size, 
                                    random_state=random_state)
            train_idx, test_idx = next(gss.split(df, groups=df[patient_id_column]))
            
        else:
            if stratify_column:
                train_idx, test_idx = next(
                        StratifiedShuffleSplit(test_size=test_size, random_state=random_state).split(
                            df, df[stratify_column])
                    )
            else:
                train_idx, test_idx = next(
                        ShuffleSplit(test_size=test_size, random_state=random_state).split(
                            df)
                    )
                

        return self.split_by_index(train_idx, test_idx)


    def split_by_index(self, train_index, test_index):
        """데이터 index를 이용하여 split하는 함수

        Args:
            train_index ([array-like of shape (n_samples,)]): 첫번째 데이터셋을 구성할 데이터의 index
            test_index ([array-like of shape (n_samples,)]): 두번째 데이터셋을 구성할 데이터의 index

        Returns:
            [Tuple of Dataset]: 분리된 2개의 Dataset
        """

        train_df = self.metadata_df.iloc[train_index, :]
        test_df = self.metadata_df.iloc[test_index, :]

        train_ds = copy.copy(self).set_metadata(train_df)
        test_ds = copy.copy(self).set_metadata(test_df)

        return train_ds, test_ds

    def append(self, new_dataset):
        self.metadata_df = pd.concat((self.metadata_df, new_dataset.metadata_df), ignore_index=True)
        return self
