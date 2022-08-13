"""
This class is (almost) directly copied from the dreamquark-ai's repo:
https://github.com/dreamquark-ai/tabnet

Therefore, ALL CREDIT to the dreamquark-ai's team
"""

from typing import Tuple

import torch
from torch import nn


class RandomObfuscator(nn.Module):
    r"""Creates and applies an obfuscation masks

    Note that the class will return a mask tensor with 1s IF the feature value
    is considered for reconstruction

    Parameters:
    ----------
    p: float
        Ratio of features that will be discarded for reconstruction
    """

    def __init__(self, p: float):
        super(RandomObfuscator, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.bernoulli(self.p * torch.ones(x.shape)).to(x.device)
        masked_input = torch.mul(1 - mask, x)
        return masked_input, mask
