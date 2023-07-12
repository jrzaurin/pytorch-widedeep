import json
import warnings
from inspect import signature
from pathlib import Path

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
from pytorch_widedeep.stream._stream_ds import StreamTextDataset, StreamWideDeepDataset, StreamImageDataset


class StreamTrainer(Trainer):
    def __init__(
            self,
            model: WideDeep, 
            objective: str,
            X_path: str,
            img_col: str,
            text_col: str,
            target_col: str,
            text_preprocessor: StreamTextPreprocessor,
            img_preprocessor: StreamImagePreprocessor,
            fetch_size: int = 5 # How many examples to load into memory at once.
        ):
        super().__init__(model, objective)
        self.X_path = X_path
        self.img_col = img_col
        self.text_col = text_col
        self.target_col = target_col
        self.text_preprocessor = text_preprocessor
        self.img_preprocessor = img_preprocessor
        self.fetch_size = fetch_size
        self.with_lds = False

    def fit(
            self, 
            batch_size: int = 32,
            n_epochs: int = 1,
            lds_weightt: Tensor = Tensor(0), #TODO: Fix this
            with_lds: bool = False
        ):
        
        self.with_lds = with_lds

        # Move into a function that constructs this similar to _build_train_dict
        # As we need to cater to different combos of each mode
        train_wd = StreamWideDeepDataset(
            self.X_path,
            self.img_col,
            self.text_col,
            self.target_col,
            self.text_preprocessor,
            self.img_preprocessor,
            self.fetch_size
        )

        # Use _trainer_utils _build_train_dict!
        train_loader = DataLoader(
            train_wd, 
            batch_size=batch_size,
            drop_last=True
        )

        # TODO: enable validation callbacks
        # eval_set = None
        # if eval_set is not None:
        #     train_loader = DataLoader(
        #         StreamTextDataset(
        #             X_val_path, 
        #             preprocessor=preprocessor, 
        #             chunksize=chunksize
        #         ), 
        #     batch_size=batch_size,
        #     drop_last=True
        # )

        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            for batch_idx, (data, targett) in enumerate(train_loader):
                # breakpoint()
                train_score, train_loss = self._train_step(
                    data, targett, batch_idx, epoch, lds_weightt
                )
                print(train_loss)
                # print_loss_and_metric(batch_idx, train_loss, train_score)
                # self.callback_container.on_batch_end(batch=batch_idx)

            # epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            # self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)

        # self.callback_container.on_train_end(epoch_logs)
        self.model.train() 