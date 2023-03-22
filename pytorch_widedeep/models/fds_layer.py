import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import Tuple, Union, Tensor, Literal, Optional
from pytorch_widedeep.utils.deeptabular_utils import (
    find_bin,
    get_kernel_window,
)


class FDSLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        granularity: int = 100,
        y_max: Optional[float] = None,
        y_min: Optional[float] = None,
        start_update: int = 0,
        start_smooth: int = 2,
        kernel: Literal["gaussian", "triang", "laplace"] = "gaussian",
        ks: int = 5,
        sigma: float = 2,
        momentum: Optional[float] = 0.9,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        """
        Feature Distribution Smoothing layer. Please, see
        [Delving into Deep Imbalanced Regression](https:/arxiv.org/abs/2102.09554)
        for details.

        :information_source: **NOTE**: this is NOT an available model per se,
         but more a utility that can be used as we run a `WideDeep` model.
         The parameters of this extra layers can be set as the class
         `WideDeep` is instantiated via the keyword arguments `fds_config`.

        :information_source: **NOTE**: Feature Distribution Smoothing is
         available when using ONLY a `deeptabular` component

        :information_source: **NOTE**: We consider this feature absolutely
        experimental and we recommend the user to not use it unless the
        corresponding [publication](https://arxiv.org/abs/2102.09554) is
        well understood

        The code here is based on the code at the
        [official repo](https://github.com/YyzHarry/imbalanced-regression)

        Parameters
        ----------
        feature_dim: int,
            input dimension size, i.e. output size of previous layer. This
            will be the dimension of the output from the `deeptabular`
            component
        granularity: int = 100,
            number of bins that the target $y$ is divided into and that will
            be used to compute the features' statistics (mean and variance)
        y_max: Optional[float] = None,
            $y$ upper limit to be considered when binning
        y_min: Optional[float] = None,
            $y$ lower limit to be considered when binning
        start_update: int = 0,
            number of _'waiting epochs' after which the FDS layer will start
            to update its statistics
        start_smooth: int = 1,
            number of _'waiting epochs' after which the FDS layer will start
            smoothing the feature distributions
        kernel: Literal["gaussian", "triang", "laplace", None] = "gaussian",
            choice of smoothing kernel
        ks: int = 5,
            kernel window size
        sigma: Union[int, float] = 2,
            if a _'gaussian'_ or _'laplace'_ kernels are used, this is the
            corresponding standard deviation
        momentum: float = 0.9,
            to train the layer the authors used a momentum update of the running
            statistics across each epoch. Set to 0.9 in the paper.
        clip_min: Optional[float] = None,
            this parameter is used to clip the ratio between the so called
            running variance and the smoothed variance, and is introduced for
            numerical stability. We leave it as optional as we did not find a
            notable improvement in our experiments. The authors used a value
            of 0.1
        clip_max: Optional[float] = None,
            same as `clip_min` but for the upper limit.We leave it as optional
            as we did not find a notable improvement in our experiments. The
            authors used a value of 10.
        """
        super(FDSLayer, self).__init__()
        assert (
            start_update + 1 < start_smooth
        ), "initial update must start at least 2 epoch before smoothing"

        self.feature_dim = feature_dim
        self.granularity = granularity
        self.y_max = y_max
        self.y_min = y_min
        self.kernel_window = torch.tensor(
            get_kernel_window(kernel, ks, sigma), dtype=torch.float32
        )
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.pred_layer = nn.Linear(feature_dim, 1)

        self._register_buffers()

    def forward(self, features, labels, epoch) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.training:
            features = self._smooth(features, labels, epoch)
            return (features, self.pred_layer(features))
        else:
            return self.pred_layer(features)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch > self.start_update:
            self.running_mean_last_epoch = self.running_mean
            self.running_var_last_epoch = self.running_var

            smoothed_mean_last_epoch_inp = F.pad(
                self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                pad=(self.half_ks, self.half_ks),
                mode="reflect",
            )
            smoothed_mean_last_epoch_weight = self.kernel_window.view(1, 1, -1).to(
                smoothed_mean_last_epoch_inp.device
            )
            self.smoothed_mean_last_epoch = (
                F.conv1d(
                    input=smoothed_mean_last_epoch_inp,
                    weight=smoothed_mean_last_epoch_weight,
                    padding=0,
                )
                .permute(2, 1, 0)
                .squeeze(1)
            )

            smoothed_var_last_epoch_inp = F.pad(
                self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                pad=(self.half_ks, self.half_ks),
                mode="reflect",
            )
            smoothed_var_last_epoch_weight = self.kernel_window.view(1, 1, -1).to(
                smoothed_var_last_epoch_inp.device
            )
            self.smoothed_var_last_epoch = (
                F.conv1d(
                    input=smoothed_var_last_epoch_inp,
                    weight=smoothed_var_last_epoch_weight,
                    padding=0,
                )
                .permute(2, 1, 0)
                .squeeze(1)
            )

    def update_running_stats(self, features, labels, epoch):
        assert self.feature_dim == features.size(
            1
        ), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(
            0
        ), "Dimensions of features and labels are not aligned!"

        if epoch == 0:
            if not self.y_max:
                self.y_max = labels.max()
            if not self.y_min:
                self.y_min = labels.min()
            bin_edges = torch.linspace(self.y_min, self.y_max, steps=self.granularity)
            self.register_buffer("bin_edges", bin_edges)

        if epoch >= self.start_update:
            left_bin_edges_indices = find_bin(
                self.bin_edges, labels.squeeze(), ret_value=False
            )
            for left_bin_edge_ind in torch.unique(left_bin_edges_indices):
                inds = (left_bin_edges_indices == left_bin_edge_ind).nonzero().squeeze()
                curr_feats = features[inds]
                curr_num_sample = curr_feats.size(0)
                curr_mean = torch.mean(curr_feats, 0)
                curr_var = torch.var(
                    curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False
                )

                self.num_samples_tracked[left_bin_edge_ind] += curr_num_sample
                if not self.momentum:
                    factor = 1 - curr_num_sample / float(
                        self.num_samples_tracked[left_bin_edge_ind]
                    )
                else:
                    factor = self.momentum
                if epoch == self.start_update:
                    factor = 0
                self.running_mean[left_bin_edge_ind] = (
                    1 - factor
                ) * curr_mean + factor * self.running_mean[left_bin_edge_ind]
                self.running_var[left_bin_edge_ind] = (
                    1 - factor
                ) * curr_var + factor * self.running_var[left_bin_edge_ind]

    def _smooth(self, features, labels, epoch):
        smoothed_features = features.detach()

        if epoch >= self.start_smooth:
            left_bin_edges_indices = find_bin(
                self.bin_edges, labels.squeeze(), ret_value=False
            )
            for left_bin_edge_ind in torch.unique(left_bin_edges_indices):
                inds = (left_bin_edges_indices == left_bin_edge_ind).nonzero().squeeze()
                smoothed_features[inds] = self._calibrate_mean_var(
                    smoothed_features[inds], left_bin_edge_ind
                )

        return smoothed_features

    def _calibrate_mean_var(self, features, left_bin_edge_ind):
        # rescaling of the data https://stats.stackexchange.com/a/46431
        m1 = self.running_mean_last_epoch[left_bin_edge_ind]
        v1 = self.running_var_last_epoch[left_bin_edge_ind]
        m2 = self.smoothed_mean_last_epoch[left_bin_edge_ind]
        v2 = self.smoothed_var_last_epoch[left_bin_edge_ind]
        if torch.sum(v1) < 1e-10:
            return features
        if (v1 == 0.0).any():
            valid = v1 != 0.0
            factor = v2[valid] / v1[valid]
            if self.clip_min and self.clip_max:
                factor = torch.clamp(factor, self.clip_min, self.clip_max)
            if features.dim() == 1:
                # the tensor has to be 2d
                features = features.unsqueeze(0)
            features[:, valid] = (features[:, valid] - m1[valid]) * torch.sqrt(
                factor
            ) + m2[valid]
            return features

        factor = v2 / v1
        if self.clip_min and self.clip_max:
            factor = torch.clamp(factor, self.clip_min, self.clip_max)

        return (features - m1) * torch.sqrt(factor) + m2

    def _register_buffers(self):
        self.register_buffer(
            "running_mean", torch.zeros(self.granularity - 1, self.feature_dim)
        )
        self.register_buffer(
            "running_var", torch.ones(self.granularity - 1, self.feature_dim)
        )
        self.register_buffer(
            "running_mean_last_epoch",
            torch.zeros(self.granularity - 1, self.feature_dim),
        )
        self.register_buffer(
            "running_var_last_epoch", torch.ones(self.granularity - 1, self.feature_dim)
        )
        self.register_buffer(
            "smoothed_mean_last_epoch",
            torch.zeros(self.granularity - 1, self.feature_dim),
        )
        self.register_buffer(
            "smoothed_var_last_epoch",
            torch.ones(self.granularity - 1, self.feature_dim),
        )
        self.register_buffer("num_samples_tracked", torch.zeros(self.granularity - 1))
