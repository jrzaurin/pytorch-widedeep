import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.callbacks import Callback
from pytorch_widedeep.training._trainer_utils import (
    save_epoch_logs,
    print_loss_and_metric,
)
from pytorch_widedeep.self_supervised_training._base_self_supervised_trainer import (
    BaseSelfSupervisedTrainer,
)


class SelfSupervisedTrainer(BaseSelfSupervisedTrainer):
    def __init__(
        self,
        model,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        loss_type: Literal["contrastive", "denoising", "both"] = "both",
        projection_head1_dims: Optional[List[int]] = None,
        projection_head2_dims: Optional[List[int]] = None,
        projection_heads_activation: str = "relu",
        cat_mlp_type: Literal["single", "multiple"] = "multiple",
        cont_mlp_type: Literal["single", "multiple"] = "multiple",
        denoise_mlps_activation: str = "relu",
        verbose: int = 1,
        seed: int = 1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_type=loss_type,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=callbacks,
            projection_head1_dims=projection_head1_dims,
            projection_head2_dims=projection_head2_dims,
            projection_heads_activation=projection_heads_activation,
            cat_mlp_type=cat_mlp_type,
            cont_mlp_type=cont_mlp_type,
            denoise_mlps_activation=denoise_mlps_activation,
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
                for batch_idx, X in zip(t, pretrain_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_loss = self._pretrain_step(X[0], batch_idx)
                    self.callback_container.on_batch_end(batch=batch_idx)
                    print_loss_and_metric(t, train_loss)

            epoch_logs = save_epoch_logs(epoch_logs, train_loss, None, "train")
            self.callback_container.on_epoch_end(epoch, epoch_logs)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        self._restore_best_weights()
        self.ss_model.train()

    def _pretrain_step(self, X_tab: Tensor, batch_idx: int):

        X = X_tab.to(self.device)

        self.optimizer.zero_grad()
        g_projs, cat_x_and_x_, cont_x_and_x_ = self.ss_model(X)
        loss = self._compute_loss(g_projs, cat_x_and_x_, cont_x_and_x_)
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return avg_loss
