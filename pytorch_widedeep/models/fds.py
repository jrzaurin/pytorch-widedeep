# all credits go to     
# `Yang, Y., Zha, K., Chen, Y. C., Wang, H., & Katabi, D. (2021).
# Delving into Deep Imbalanced Regression. arXiv preprint arXiv:2102.09554.`
# and their `implementation
# <https://github.com/YyzHarry/imbalanced-regression>`

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            else:
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        return features