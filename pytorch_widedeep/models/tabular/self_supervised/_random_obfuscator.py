import torch
from torch import nn


class RandomObfuscator(nn.Module):
    def __init__(self, p):
        super(RandomObfuscator, self).__init__()
        self.p = p

    def forward(self, x):
        mask = torch.bernoulli(self.p * torch.ones(x.shape)).to(x.device)
        masked_input = torch.mul(1 - mask, x)
        return masked_input, mask
