import json
import warnings
from inspect import signature
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch import nn
from scipy.sparse import csc_matrix
from torchmetrics import Metric as TorchMetric
from torch.utils.data import DataLoader

from pytorch_widedeep.losses import ZILNLoss
from pytorch_widedeep.metrics import Metric
from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
    WideDeep,
    Optimizer,
    Transforms,
    LRScheduler,
)
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.dataloaders import DataLoaderDefault
from pytorch_widedeep.initializers import Initializer
from pytorch_widedeep.training._finetune import FineTune
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from pytorch_widedeep.training._base_trainer import BaseTrainer
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    wd_train_val_split,
    print_loss_and_metric,
)
from torchmetrics import Metric as TorchMetric
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.initializers import Initializer
from pytorch_widedeep.metrics import Metric
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.wdtypes import Dict, LRScheduler, List, Module, Optimizer, Optional, Transforms, Union, WideDeep

from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor
from pytorch_widedeep.stream.preprocessing.image_preprocessor import StreamImagePreprocessor
from pytorch_widedeep.stream._stream_ds import StreamWideDeepDataset


class StreamTrainer(Trainer):
    def __init__(
            self,
            model: WideDeep, 
            objective: str,
            custom_loss_function: Optional[nn.Module] = None,
            text_preprocessor: StreamTextPreprocessor = None,
            img_preprocessor: StreamImagePreprocessor = None,
            fetch_size: int = 5 # How many examples to load into memory at once.
    ):
        super().__init__(
            model=model, 
            objective=objective,
            custom_loss_function=custom_loss_function
        )
        self.text_preprocessor = text_preprocessor
        self.img_preprocessor = img_preprocessor
        self.fetch_size = fetch_size
        self.with_lds = False

    def fit(
            self, 
            X_train_path: Optional[str] = None,
            X_val_path: Optional[str] = None,
            target_col: str = None,
            img_col: str = None,
            text_col: str = None,
            batch_size: int = 32,
            n_epochs: int = 1,
            validation_freq: int = 1,
            lds_weightt: Tensor = Tensor(0), #TODO: Fix this
            with_lds: bool = False
        ):
        
        self.with_lds = with_lds

        train_set = StreamWideDeepDataset(
            X_path=X_train_path,
            target_col=target_col,
            img_col=img_col,
            text_col=text_col,
            text_preprocessor=self.text_preprocessor,
            img_preprocessor=self.img_preprocessor,
            fetch_size=self.fetch_size
        )

        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size,
            drop_last=False
        )

        train_steps = 0
        for _, _ in enumerate(train_loader):
            train_steps += 1 

        if X_val_path is not None:
            eval_set = StreamWideDeepDataset(
                X_path=X_val_path,
                target_col=target_col,
                img_col=img_col,
                text_col=text_col,
                text_preprocessor=self.text_preprocessor,
                img_preprocessor=self.img_preprocessor,
                fetch_size=self.fetch_size
            )
            eval_loader = DataLoader(
                eval_set, 
                batch_size=batch_size,
                drop_last=True
            )
            
            eval_steps = 0
            for _, _ in enumerate(train_loader):
                eval_steps += 1 
        
        self.callback_container.on_train_begin(
            {"batch_size": batch_size, "train_steps": train_steps, "n_epochs": n_epochs}
        )
            
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0

            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, targett) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_score, train_loss = self._train_step(
                        data, targett, batch_idx, epoch, lds_weightt
                    )
                    print_loss_and_metric(t, train_loss, train_score)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            if eval_set is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data, targett) in zip(v, eval_loader):
                        v.set_description("valid")
                        val_score, val_loss = self._eval_step(data, targett, i)
                        print_loss_and_metric(v, val_loss, val_score)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, val_score, "val")

                if self.reducelronplateau:
                    if self.reducelronplateau_criterion == "loss":
                        on_epoch_end_metric = val_loss
                    else:
                        on_epoch_end_metric = val_score[
                            self.reducelronplateau_criterion
                        ]
            else:
                if self.reducelronplateau:
                    raise NotImplementedError(
                        "ReduceLROnPlateau scheduler can be used only with validation data."
                    )
            self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

            if self.model.with_fds:
                self._update_fds_stats(train_loader, epoch)

        self.callback_container.on_train_end(epoch_logs)
        if self.model.is_tabnet:
            self._compute_feature_importance(train_loader)
        self._restore_best_weights()
        self.model.train()
