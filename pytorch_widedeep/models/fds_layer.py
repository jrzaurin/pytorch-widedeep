import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_widedeep.wdtypes import *  # noqa: F403
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
        """Feature Distribution Smoothing layer. Layer keeps track of last epoch mean
        and variance for each feature. The feautres are bucket-ed into bins based on
        their target/label value. Target/label values are binned using histogram with
        same width bins, their number is based on granularity parameter and start/end
        edge on y_max/y_min values. Mean and variance are smoothed using convolution
        with kernel function(gaussian by default). Output of the layer are features
        values adjusted to their smoothed mean and variance. The layer is turned on
        only during training, off during prediction/evaluation.

        Adjusted code from `<https://github.com/YyzHarry/imbalanced-regression>`
        For more infomation about please read the paper) :

        `Yang, Y., Zha, K., Chen, Y. C., Wang, H., & Katabi, D. (2021).
        Delving into Deep Imbalanced Regression. arXiv preprint arXiv:2102.09554.`

        Parameters
        ----------
        feature_dim: int,
            input dimension size, i.e. output size of previous layer
        granularity: int = 100,
            number of bins in histogram used for storing feature mean and variance
            values per label
        y_max: Optional[float] = None,
            option to restrict the histogram bins by upper label limit
        y_min: Optional[float] = None,
            option to restrict the histogram bins by lower label limit
        start_update: int = 0,
            epoch after which FDS layer starts to update its statistics
        start_smooth: int = 1,
            epoch after which FDS layer starts to smooth feautture distributions
        kernel: Literal["gaussian", "triang", "laplace", None] = "gaussian",
            choice of kernel for Feature Distribution Smoothing
        ks: int = 5,
            LDS kernel window size
        sigma: Union[int,float] = 2,
            standard deviation of ['gaussian','laplace'] kernel for LDS
        momentum: float = 0.9,
            factor parameter for running mean and variance
        clip_min: Optional[float] = None,
            original value = 0.1, author note: clipping is for numerical stability,
            if some bins contain a very small number of samples, the variance
            estimation may not be stable
        clip_max: Optional[float] = None,
            original value = 10, see note for clip_min
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
