import json
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Tensor,
    Optional,
    Optimizer,
    LRScheduler,
    ModelWithoutAttention,
    DecoderWithoutAttention,
)
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    print_loss_and_metric,
)
from pytorch_widedeep.self_supervised_training._base_encoder_decoder_trainer import (
    BaseEncoderDecoderTrainer,
)


class EncoderDecoderTrainer(BaseEncoderDecoderTrainer):
    r"""This class implements an Encoder-Decoder self-supervised 'routine'
    inspired by
    [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442).
    See Figure 1 above.

    Parameters
    ----------
    encoder: ModelWithoutAttention,
        An instance of a `TabMlp`, `TabResNet` or `TabNet` model
    decoder: Optional[DecoderWithoutAttention] = None,
        An instance of  a `TabMlpDecoder`, `TabResNetDecoder` or
        `TabNetDecoder` model. if `None` the decoder will be automatically
        built as a '_simetric_' model to the Encoder
    masked_prob: float = 0.2,
        Indicates the fraction of elements in the embedding tensor that will
        be masked and hence used for reconstruction
    optimizer: Optional[Optimizer] = None,
        An instance of Pytorch's `Optimizer` object (e.g. `torch.optim.Adam
        ()`). if no optimizer is passed it will default to `AdamW`.
    lr_scheduler: Optional[LRScheduler] = None,
        An instance of Pytorch's `LRScheduler` object
        (e.g `torch.optim.lr_scheduler.StepLR(opt, step_size=5)`).
    callbacks: Optional[List[Callback]] = None,
        List with `Callback` objects. The three callbacks available in
        `pytorch-widedeep` are: `LRHistory`, `ModelCheckpoint` and
        `EarlyStopping`. This can also be a custom callback. See
        `pytorch_widedeep.callbacks.Callback` or the Examples folder in the
        repo.
    verbose: int, default=1
        Setting it to 0 will print nothing during training.
    seed: int, default=1
        Random seed to be used internally for train_test_split

    Other Parameters
    ----------------
    **kwargs: dict
        Other infrequently used arguments that can also be passed as kwargs are:

        - **device**: `str`<br/>
            string indicating the device. One of _'cpu'_ or _'gpu'_

        - **num_workers**: `int`<br/>
            number of workers to be used internally by the data loaders

        - **reducelronplateau_criterion**: `str`
            This sets the criterion that will be used by the lr scheduler to
            take a step: One of _'loss'_ or _'metric'_. The ReduceLROnPlateau
            learning rate is a bit particular.

    """

    def __init__(
        self,
        encoder: ModelWithoutAttention,
        decoder: Optional[DecoderWithoutAttention] = None,
        masked_prob: float = 0.2,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1,
        seed: int = 1,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            masked_prob=masked_prob,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=callbacks,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )

    def pretrain(
        self,
        X_tab: np.ndarray,
        X_tab_val: Optional[np.ndarray] = None,
        val_split: Optional[float] = None,
        validation_freq: int = 1,
        n_epochs: int = 1,
        batch_size: int = 32,
    ):
        r"""Pretrain method. Can also be called using `.fit(<same_args>)`

        Parameters
        ----------
        X_tab: np.ndarray,
            tabular dataset
        X_tab_val: np.ndarray, Optional, default = None
            validation data
        val_split: float, Optional. default=None
            An alterative to passing the validation set is to use a train/val
            split fraction via `val_split`
        validation_freq: int, default=1
            epochs validation frequency
        n_epochs: int, default=1
            number of epochs
        batch_size: int, default=32
            batch size
        """

        self.batch_size = batch_size

        train_set, eval_set = self._train_eval_split(X_tab, X_tab_val, val_split)
        train_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, num_workers=self.num_workers
        )
        train_steps = len(train_loader)
        if eval_set is not None:
            eval_loader = DataLoader(
                dataset=eval_set,
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            eval_steps = len(eval_loader)

        self.callback_container.on_train_begin(
            {
                "batch_size": batch_size,
                "train_steps": train_steps,
                "n_epochs": n_epochs,
            }
        )
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, X in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_loss = self._train_step(X[0], batch_idx)
                    self.callback_container.on_batch_end(batch=batch_idx)
                    print_loss_and_metric(t, train_loss)

            epoch_logs = save_epoch_logs(epoch_logs, train_loss, None, "train")

            on_epoch_end_metric = None
            if eval_set is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for batch_idx, X in zip(v, eval_loader):
                        v.set_description("valid")
                        val_loss = self._eval_step(X[0], batch_idx)
                        print_loss_and_metric(v, val_loss)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, None, "val")
                on_epoch_end_metric = val_loss
            else:
                if self.reducelronplateau:
                    raise NotImplementedError(
                        "ReduceLROnPlateau scheduler can be used only with validation data."
                    )

            self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        self._restore_best_weights()
        self.ed_model.train()

    def fit(
        self,
        X_tab: np.ndarray,
        X_tab_val: Optional[np.ndarray] = None,
        val_split: Optional[float] = None,
        validation_freq: int = 1,
        n_epochs: int = 1,
        batch_size: int = 32,
    ):
        return self.pretrain(
            X_tab, X_tab_val, val_split, validation_freq, n_epochs, batch_size
        )

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "ed_model.pt",
    ):
        r"""Saves the model, training and evaluation history (if any) to disk

        Parameters
        ----------
        path: str
            path to the directory where the model and the feature importance
            attribute will be saved.
        save_state_dict: bool, default = False
            Boolean indicating whether to save directly the model or the
            model's state dictionary
        model_filename: str, Optional, default = "ed_model.pt"
            filename where the model weights will be store
        """
        save_dir = Path(path)
        history_dir = save_dir / "history"
        history_dir.mkdir(exist_ok=True, parents=True)

        # the trainer is run with the History Callback by default
        with open(history_dir / "train_eval_history.json", "w") as teh:
            json.dump(self.history, teh)  # type: ignore[attr-defined]

        has_lr_history = any(
            [clbk.__class__.__name__ == "LRHistory" for clbk in self.callbacks]
        )
        if self.lr_scheduler is not None and has_lr_history:
            with open(history_dir / "lr_history.json", "w") as lrh:
                json.dump(self.lr_history, lrh)  # type: ignore[attr-defined]

        model_path = save_dir / model_filename
        if save_state_dict:
            torch.save(self.ed_model.state_dict(), model_path)
        else:
            torch.save(self.ed_model, model_path)

    def explain(self, X_tab: np.ndarray, save_step_masks: bool = False):
        raise NotImplementedError(
            "The 'explain' is currently not implemented for Self Supervised Pretraining"
        )

    def _train_step(self, X_tab: Tensor, batch_idx: int) -> float:
        X = X_tab.to(self.device)

        self.optimizer.zero_grad()
        x_embed, x_embed_rec, mask = self.ed_model(X)
        loss = self.loss_fn(x_embed, x_embed_rec, mask)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return avg_loss

    def _eval_step(self, X_tab: Tensor, batch_idx: int) -> float:
        self.ed_model.eval()

        with torch.no_grad():
            X = X_tab.to(self.device)

            x_embed, x_embed_rec, mask = self.ed_model(X)
            loss = self.loss_fn(x_embed, x_embed_rec, mask)

            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return avg_loss

    def _train_eval_split(
        self,
        X: np.ndarray,
        X_tab_val: Optional[np.ndarray] = None,
        val_split: Optional[float] = None,
    ) -> Tuple[TensorDataset, Optional[TensorDataset]]:
        if X_tab_val is not None:
            train_set = TensorDataset(torch.from_numpy(X))
            eval_set = TensorDataset(torch.from_numpy(X_tab_val))
        elif val_split is not None:
            X_tr, X_tab_val = train_test_split(
                X, test_size=val_split, random_state=self.seed
            )
            train_set = TensorDataset(torch.from_numpy(X_tr))
            eval_set = TensorDataset(torch.from_numpy(X_tab_val))
        else:
            train_set = TensorDataset(torch.from_numpy(X))
            eval_set = None

        return train_set, eval_set
