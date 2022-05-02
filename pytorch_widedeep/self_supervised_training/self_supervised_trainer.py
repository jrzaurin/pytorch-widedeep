import numpy as np

import torch
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from pytorch_widedeep.losses import InfoNCELoss, DenoisingLoss, ContrastiveLoss
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.training._trainer_utils import save_epoch_logs
from pytorch_widedeep.self_supervised_training._augmentations import (
    mix_up,
    cut_mix,
)
from pytorch_widedeep.self_supervised_training._base_self_supervised_trainer import (
    BaseSelfSupervisedTrainer,
)


class SelfSupervisedTrainer(BaseSelfSupervisedTrainer):
    def __init__(
        self,
        model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        callbacks: Optional[List[Callback]],
        verbose: int,
        seed: int,
        **kwargs,
    ):
        super().__init__(
            model=model,
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
        n_epochs: int = 1,
        batch_size: int = 32,
    ):

        self.batch_size = batch_size

        pretrain_loader = DataLoader(
            dataset=TensorDataset(torch.from_numpy(X_tab)),
            batch_size=batch_size,
            num_workers=self.num_workers,
        )
        train_steps = len(pretrain_loader)

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
                for batch_idx, (X, y) in zip(t, pretrain_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_loss = self._pretrain_step(X, batch_idx)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, None, "train")
            self.callback_container.on_epoch_end(epoch, epoch_logs)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        self._restore_best_weights()
        self.model.train()

    def _pretrain_step(self, X_tab: np.ndarray, batch_idx: int):

        X = X_tab.to(self.device)

        self.optimizer.zero_grad()

        encoded = self.model(X)
        cut_mixed_x = cut_mix(X)
        cut_mixed_x_embed = self.model.cat_and_cont_embed(cut_mixed_x)
        cut_mixed_x_mixed_up_embed = mix_up(cut_mixed_x_embed)
        encoded_ = self.model.transformer_blks(cut_mixed_x_mixed_up_embed)
        proj_encoded = self.projection_head1(encoded)
        proj_encoded_ = (
            self.projection_head2(encoded_)
            if self.projection_head2 is not None
            else self.projection_head1(encoded_)
        )

        loss = self.loss(proj_encoded, proj_encoded_)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return avg_loss
