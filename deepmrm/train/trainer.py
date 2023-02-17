import time
from pathlib import Path

import torch.optim
from torchmetrics import MetricCollection
import numpy as np

from mstorch.utils.cuda_memory import recursive_copy_to_device
from mstorch.enums import PartitionType
from deepmrm.train.monitor import TrainMonitor

from tqdm import tqdm

TORCH_MODEL_FILE_EXT = 'pth'


class MetricManager:
    
    def __init__(self, task):
        super().__init__()
        self.task = task
        metric_sets = {
            PartitionType.TRAIN: dict(),
            PartitionType.VALIDATION: dict(),
        }
        for t in self.task:
            metric_sets[PartitionType.TRAIN][t] = MetricCollection([], prefix=f'{t.name}_', postfix='_train')
            metric_sets[PartitionType.VALIDATION][t] = MetricCollection([], prefix=f'{t.name}_', postfix='_val')
            
        self.metric_sets = metric_sets

    def _generator(self):
        for partition, metric_sets in self.metric_sets.items():
            for task_name, metric_set in metric_sets.items():            
                yield metric_set

    def get_metrics(self, partition_type):
        return self.metric_sets[partition_type]
    
    def add_metrics(self, task, metrics):
        self.metric_sets[PartitionType.TRAIN][task].add_metrics(metrics.clone())
        self.metric_sets[PartitionType.VALIDATION][task].add_metrics(metrics.clone())

    def to(self, device):
        for metric_set in self._generator():
            metric_set.to(device)
        return self

    def compute(self):
        computed = dict()
        for metric_set in self._generator():
            computed.update(metric_set.compute())
        return computed

    def reset(self):
        for metric_set in self._generator():
            metric_set.reset()

    def __repr__(self) -> str:
        repr_str = '\n'.join([
            f'{metrics} for {task}' for task, metrics in self.get_metrics(PartitionType.TRAIN).items()
        ])
        return repr_str


class BaseTrainer(object):

    def __init__(self, 
                 data_manager,
                 save_dir,
                 logger,
                 batch_size=32,
                 run_copy_to_device=True,
                 gpu_index=0):

        self.data_manager = data_manager
        self.logger = logger
        self.gpu_index = gpu_index
        self.epoch_index = -1

        self.loss = None
        self.model = None
        self.optimizer = None
        self.optimizer_schedulers = []

        self.num_epochs = 0
        self.num_epochs = 1
        self.batch_size = batch_size
        
        self.set_save_dir(save_dir)
        self.monitor = TrainMonitor(patience=50)
        self.early_stop = False
        self.metric_manager = MetricManager(self.task)
        self.add_metrics = self.metric_manager.add_metrics
        self.run_copy_to_device = run_copy_to_device

    @property
    def task(self):
        return self.data_manager.task

    @property
    def use_gpu(self):
        return self.gpu_index >= 0

    @property
    def device(self):
        if self.use_gpu:
            return torch.device("cuda:{}".format(self.gpu_index))
        return torch.device("cpu")

    def set_gpu_index(self, gpu_index):
        assert gpu_index < torch.cuda.device_count() 
        self.gpu_index = gpu_index
        return self

    def set_data_manager(self, data_mgr):
        self.data_manager = data_mgr
        return self

    def set_model(self, model):
        self.model = model
        
        return self

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        self.optimizer = optimizer
        return self

    def set_monitor(self, monitor):
        self.monitor = monitor
        return self

    def set_scheduler(self, scheduler):
        if not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise ValueError(f'{type(scheduler)} should be an instance of torch.optim.lr_scheduler._LRScheduler')
        self.optimizer_schedulers.append(scheduler)
        return self


    def train(self, num_epochs, batch_size):

        self._start_training(num_epochs, batch_size)
        train_metrics = self.metric_manager.get_metrics(PartitionType.TRAIN)
        val_metrics = self.metric_manager.get_metrics(PartitionType.VALIDATION)

        while not self.is_done():
            # prepare the start of new epoch
            self._on_epoch_start()

            # training step
            self.model.train()
            batch_iterator = enumerate(tqdm(self.data_loaders[PartitionType.TRAIN], position=0,colour='blue'))
            for batch_idx, batched_samples_ in batch_iterator:
                batch_size_ = len(next(iter(batched_samples_.values())))
                batched_samples = self._copy_to_device(batched_samples_)

                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(batched_samples, metrics=train_metrics)
                loss.backward() # back prop
                self.optimizer.step() # grad
                self.train_losses.append(loss.item()*batch_size_)

                if (batch_idx+1) % 50 == 0:
                    cur_train_loss = np.sum(self.train_losses) / (self.batch_size*(batch_idx+1))
                    self.logger.debug(
                        f'[epoch={self.epoch_index}] batch={batch_idx+1}, train loss:{cur_train_loss}')

            # evaluation step
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batched_samples_ in enumerate(tqdm(self.data_loaders[PartitionType.VALIDATION], position=0)):
                    batch_size_ = len(next(iter(batched_samples_.values())))
                    batched_samples = self._copy_to_device(batched_samples_)
                    loss = self.model.compute_loss(batched_samples, metrics=val_metrics)
                    self.val_losses.append(loss.item()*batch_size_)

            # finalize epoch
            self._on_epoch_end()

        self._complete_training()

        # [TODO] remove model from gpu memory after training...

    def is_done(self):
        return self.early_stop or (self.epoch_index >= self.num_epochs)

    def _start_training(self, num_epochs, batch_size):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epoch_index = 0
        self.min_val_loss = np.Inf

        transform = self.data_manager.datasets[PartitionType.TEST].transform
        self.model.set_transform(transform)

        self.logger.info(f"Start training {self.model.name} on {self.device}. max epochs: {self.num_epochs}, batch size: {self.batch_size}")
        self.logger.info(f"Optimizer:\n{self.optimizer}")
        self.logger.info(f"Optimizer schedulers: {self.optimizer_schedulers}")
        self.logger.info(f"Metrics:\n{self.metric_manager}")
        
        # move model and metrics to target device
        self.model = self.model.to(self.device)
        self.metric_manager = self.metric_manager.to(self.device)
        self.logger.debug('model & metrics have been moved to {}'.format(self.device))
        self.monitor.init()

        # prepare data loaders
        self.data_loaders = dict()
        for key in PartitionType:
            self.data_loaders[key] = \
                self.data_manager.get_dataloader(key, batch_size=batch_size)

    def _complete_training(self):
        self.logger.info(f'Complete training for {self.model.name}')
        history_df = self.monitor.get_history_df()
        csv_path = self.save_dir / '{}_train_history.csv'.format(self.model.name)
        history_df.to_csv(csv_path)
        self.logger.debug('Save train history in {}'.format(csv_path))


    def _on_epoch_start(self):
        
        # Reset metrics for next epoch
        self.metric_manager.reset()
        self.train_losses = []
        self.val_losses = []

        # Setup new epoch
        self.epoch_index += 1
        self.epoch_start_time = time.perf_counter()
        self.logger.info('Start epoch {}'.format(self.epoch_index))

    def _on_epoch_end(self):
        
        self.epoch_end_time = time.perf_counter()
        for scheduler in self.optimizer_schedulers:
            scheduler.step()
        
        train_ds = self.data_manager.get_dataset(PartitionType.TRAIN)
        val_ds = self.data_manager.get_dataset(PartitionType.VALIDATION)

        # compute and update metrics for current epoch
        monitoring_metrics = {
            'train_loss': np.sum(self.train_losses) / len(train_ds),
            'val_loss': np.sum(self.val_losses) / len(val_ds),
            'elapsed': self.epoch_end_time - self.epoch_start_time,
        }
        monitoring_metrics.update(self.metric_manager.compute())

        should_stop, should_save, msg = self.monitor(monitoring_metrics)
        self.early_stop = should_stop
        self.logger.info(f'[epoch={self.epoch_index}] completed')
        if msg is not None and len(msg) > 0:
            self.logger.info(msg)
        self.logger.info(self.monitor.history[-1])

        ## save the model if validation loss has decreased
        if should_save:
            self.logger.info(f'[epoch={self.epoch_index}] Saving model')
            self.save_model()

        # if self.epoch_index == 1:
        #     self.logger.info(f'[epoch={self.epoch_index}] Freeze CNN')
        #     self.model.freeze_cnn()
    
    def _copy_to_device(self, batched_samples):
        if self.run_copy_to_device and self.use_gpu:
            batched_samples = recursive_copy_to_device(
                                batched_samples, 
                                self.device)
        
        return batched_samples

    def set_save_dir(self, save_dir, exist_ok=True):
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.logger.info(
                f'A directory for saving model({self.save_dir}) has been created.')

    def save_model(self):
        model_path = self.model_save_path
        torch.save(self.model, model_path)
        self.logger.debug(f'[epoch={self.epoch_index}] trained model has been saved to {model_path}')

    @property
    def model_save_path(self):
        model_path = self.save_dir / '{}.{}'.format(self.model.name, TORCH_MODEL_FILE_EXT)
        return model_path

    def load_model(self):
        model_path = self.model_save_path
        self.logger.debug('Load the saved model from {}'.format(model_path))
        return torch.load(model_path)

        



        


