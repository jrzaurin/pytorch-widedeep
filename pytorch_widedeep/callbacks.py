'''
Code here is mostly based on the code from the torchsample and Keras
'''
import numpy as np
import os
import time
import shutil
import datetime
import warnings
import torch

from torch import nn
from tqdm import tqdm
from copy import deepcopy
from .wdtypes import *


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks:Optional[List]=None, queue_length:int=10):
        instantiated_callbacks = []
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type): instantiated_callbacks.append(callback())
                else: instantiated_callbacks.append(callback)
        self.callbacks = [c for c in instantiated_callbacks]
        self.queue_length = queue_length

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model:nn.Module):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs:Optional[Dict]=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs:Optional[Dict]=None):
        logs = logs or {}
        # logs['final_loss'] = self.model.history.epoch_losses[-1],
        # logs['best_loss'] = min(self.model.history.epoch_losses),
        # logs['stop_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model:nn.Module):
        self.model = model

    def on_epoch_begin(self, epoch:int, logs:Optional[Dict]=None):
        pass

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        pass

    def on_batch_begin(self, batch:int, logs:Optional[Dict]=None):
        pass

    def on_batch_end(self, batch:int, logs:Optional[Dict]=None):
        pass

    def on_train_begin(self, logs:Optional[Dict]=None):
        pass

    def on_train_end(self, logs:Optional[Dict]=None):
        pass


class History(Callback):
    """
    Callback that records events into a `History` object.
    """

    def on_train_begin(self, logs:Optional[Dict]=None):
        self.epoch = []
        self._history = {}

    def on_epoch_begin(self, epoch:int, logs:Optional[Dict]=None):
        # avoid mutation during epoch run
        logs = deepcopy(logs) or {}
        for k, v in logs.items():
            self._history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self._history.setdefault(k, []).append(v)


class LRHistory(Callback):
    def __init__(self, n_epochs):
        super(LRHistory, self).__init__()
        self.n_epochs = n_epochs

    def on_epoch_begin(self, epoch:int, logs:Optional[Dict]=None):
        if epoch==0 and self.model.lr_scheduler:
            self.model.lr_history = {}
            if self.model.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
                for model_name, opt in self.model.optimizer._optimizers.items():
                    if model_name in self.model.lr_scheduler._schedulers:
                        for group_idx, group in enumerate(opt.param_groups):
                            self.model.lr_history.setdefault(
                                ("_").join(['lr', model_name, str(group_idx)]),[]
                                ).append(group['lr'])
            elif not self.model.cyclic:
                for group_idx, group in enumerate(self.model.optimizer.param_groups):
                    self.model.lr_history.setdefault(
                        ("_").join(['lr', str(group_idx)]),[]).append(group['lr'])

    def on_batch_end(self, batch:int, logs:Optional[Dict]=None):
        if self.model.lr_scheduler:
            if self.model.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
                for model_name, opt in self.model.optimizer._optimizers.items():
                    if model_name in self.model.lr_scheduler._schedulers:
                        if 'cycl' in self.model.lr_scheduler._schedulers[model_name].__class__.__name__.lower():
                            for group_idx, group in enumerate(opt.param_groups):
                                self.model.lr_history.setdefault(
                                    ("_").join(['lr', model_name, str(group_idx)]),[]
                                    ).append(group['lr'])
            elif self.model.cyclic:
                for group_idx, group in enumerate(self.model.optimizer.param_groups):
                    self.model.lr_history.setdefault(
                        ("_").join(['lr', str(group_idx)]),[]).append(group['lr'])

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        if epoch != (self.n_epochs-1) and self.model.lr_scheduler:
            if self.model.lr_scheduler.__class__.__name__ == 'MultipleLRScheduler':
                for model_name, opt in self.model.optimizer._optimizers.items():
                    if model_name in self.model.lr_scheduler._schedulers:
                        if 'cycl' not in self.model.lr_scheduler._schedulers[model_name].__class__.__name__.lower():
                            for group_idx, group in enumerate(opt.param_groups):
                                self.model.lr_history.setdefault(
                                    ("_").join(['lr', model_name, str(group_idx)]),
                                    []).append(group['lr'])
            elif not self.model.cyclic:
                for group_idx, group in enumerate(self.model.optimizer.param_groups):
                    self.model.lr_history.setdefault(
                        ("_").join(['lr', str(group_idx)]),[]).append(group['lr'])


class ModelCheckpoint(Callback):
    """
    Save the model after every epoch.
    """

    def __init__(self, filepath:str, monitor:str='val_loss', verbose:int=0,
                 save_best_only:bool=False, mode:str='auto', period:int=1,
                 max_save:int=-1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.max_save = max_save

        root_dir = ('/').join(filepath.split("/")[:-1])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if self.max_save > 0:
            self.old_files = []

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = '{}_{}'.format(self.filepath, epoch+1)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
                        if self.max_save > 0:
                            if len(self.old_files) == self.max_save:
                                try:
                                    os.remove(self.old_files[0])
                                except:
                                    pass
                                self.old_files = self.old_files[1:]
                            self.old_files.append(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)
                if self.max_save > 0:
                    if len(self.old_files) == self.max_save:
                        try:
                            os.remove(self.old_files[0])
                        except:
                            pass
                        self.old_files = self.old_files[1:]
                    self.old_files.append(filepath)


class EarlyStopping(Callback):
    """
    Stop training when a monitored quantity has stopped improving.
    """

    def __init__(self, monitor:str='val_loss', min_delta:int=0, patience:int=10,
        verbose:int=0,mode:str='auto', baseline:Optional[float]=None,
        restore_best_weights:bool=False):

        super(EarlyStopping,self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.state_dict = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs:Optional[Dict]=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch:int, logs:Optional[Dict]=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.state_dict = self.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.early_stop = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.load_state_dict(self.state_dict)

    def on_train_end(self, logs:Optional[Dict]=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn('Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value