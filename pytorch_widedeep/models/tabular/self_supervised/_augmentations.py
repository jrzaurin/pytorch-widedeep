"""
The following two functions are based on those in the SAINT repo
(https://github.com/somepago/saint) and code here:

- CutMix: https://github.com/clovaai/CutMix-PyTorch
- MixUp: https://github.com/facebookresearch/mixup-cifar10
"""

import numpy as np
import torch

from pytorch_widedeep.wdtypes import Tensor


def cut_mix(x: Tensor, lam: float = 0.8) -> Tensor:
    batch_size = x.size()[0]

    mask = torch.from_numpy(np.random.choice(2, (x.shape), p=[lam, 1 - lam])).to(
        x.device
    )

    rand_idx = torch.randperm(batch_size).to(x.device)

    x_ = x[rand_idx].clone()

    x_[mask == 0] = x[mask == 0]

    return x_


def mix_up(p: Tensor, lam: float = 0.8) -> Tensor:
    batch_size = p.size()[0]

    rand_idx = torch.randperm(batch_size).to(p.device)

    p_ = lam * p + (1 - lam) * p[rand_idx, ...]

    return p_
