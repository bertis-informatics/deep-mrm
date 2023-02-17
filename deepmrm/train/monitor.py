import numpy as np
import pandas as pd
import torch

class TrainMonitor(object):
    """Monitor a metric and stop training when it stops improving"""
    
    order_dict = {
        'min': "<",
        'max': ">",
    }

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=100, 
                 mode='min', 
                 stopping_threshold=None,
                 divergence_threshold=None):
        """
        Args:
            monitor: quantity to be monitored.
            min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
                change of less than `min_delta`, will count as no improvement.
            patience: number of checks with no improvement
                after which training will be stopped. Under the default configuration, one check happens after
                every training epoch. However, the frequency of validation can be modified by setting various parameters on
                the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

                .. note::

                    It must be noted that the patience parameter counts the number of validation checks with
                    no improvement, and not the number of training epochs. Therefore, with parameters
                    ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                    epochs before being stopped.
            mode: optimization mode which is either ``'min'`` or ``'max'``.
            stopping_threshold: Stop training immediately once the monitored quantity reaches this threshold.
            divergence_threshold: Stop training as soon as the monitored quantity becomes worse than this threshold.
        """

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.wait_count = 0
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.history = list()

        if mode == 'min':
            self.best_score = np.inf
            self.min_delta = min_delta * -1
            self.is_improved = lambda new, old : new < old
        elif mode == 'max':
            self.best_score = -np.inf
            self.min_delta = min_delta
            self.is_improved = lambda new, old : new > old
        else:
            raise NotImplementedError()

    def init(self):
        self.wait_count = 0
        self.history = list()

    def __call__(self, metrics):
        
        # record metrics
        for metric_key, metric_val in metrics.items():
            if torch.is_tensor(metric_val):
                metric_val = metric_val.item()
                metrics[metric_key] = metric_val

        self.history.append(metrics)

        should_stop = False
        should_save = False
        current = metrics[self.monitor]
        msg = ''
        
        if self.stopping_threshold is not None and self.is_improved(current, self.stopping_threshold):
            should_stop = True
            should_save = True
            msg = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.is_improved(self.divergence_threshold, current):
            should_stop = True
            msg = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.is_improved(current - self.min_delta, self.best_score):
            should_stop = False
            should_save = True
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                msg = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, should_save, msg


    def get_history_df(self):
        hitstory_df = pd.DataFrame.from_dict(self.history, dtype=np.float64)
        hitstory_df.index.name = 'epoch'
        return hitstory_df