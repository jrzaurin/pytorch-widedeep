# Losses

`pytorch-widedeep` accepts a number of losses and objectives that can be
passed to the `Trainer` class via the parameter `objective`
(see `pytorch-widedeep.training.Trainer`). For most cases the loss function
that `pytorch-widedeep` will use internally is already implemented in
Pytorch.

In addition, `pytorch-widedeep` implements a series of  "custom" loss
functions. These are described below for completion since, as mentioned
before, they are used internally by the `Trainer`. Of course, onen could
always use them on their own and can be imported as:

``
from pytorch_widedeep.losses import FocalLoss
``

---
:information_source: **NOTE**:  Losses in this module expect the predictions
 and ground truth to have the same dimensions for regression and binary
 classification problems $(N_{samples}, 1)$. In the case of multiclass
 classification problems the ground truth is expected to be a 1D tensor with
 the corresponding classes. See Examples below
---

::: pytorch_widedeep.losses.MSELoss

::: pytorch_widedeep.losses.MSLELoss

::: pytorch_widedeep.losses.RMSELoss

::: pytorch_widedeep.losses.RMSLELoss

::: pytorch_widedeep.losses.QuantileLoss

::: pytorch_widedeep.losses.FocalLoss

::: pytorch_widedeep.losses.BayesianSELoss

::: pytorch_widedeep.losses.TweedieLoss

::: pytorch_widedeep.losses.ZILNLoss

::: pytorch_widedeep.losses.L1Loss

::: pytorch_widedeep.losses.FocalR_L1Loss

::: pytorch_widedeep.losses.FocalR_MSELoss

::: pytorch_widedeep.losses.FocalR_RMSELoss

::: pytorch_widedeep.losses.HuberLoss

::: pytorch_widedeep.losses.InfoNCELoss

::: pytorch_widedeep.losses.DenoisingLoss

::: pytorch_widedeep.losses.EncoderDecoderLoss
