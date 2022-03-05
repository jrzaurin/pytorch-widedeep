Losses
======

``pytorch-widedeep`` accepts a number of losses and objectives that can be
passed to the ``Trainer`` class via the parameter ``objective``
(see ``pytorch-widedeep.training.Trainer``). For most cases the loss function
that ``pytorch-widedeep`` will use internally is already implemented in
Pytorch.

In addition, ``pytorch-widedeep`` implements a series of  "custom" loss
functions. These are described below for completion since, as mentioned
before, they are used internally by the ``Trainer``. Of course, onen could
always use them on their own and can be imported as:

.. code-block:: python

	from pytorch_widedeep.losses import FocalLoss

.. note:: Losses in this module expect the predictions and ground truth to have the
	same dimensions for regression and binary classification problems
	:math:`(N_{samples}, 1)`. In the case of multiclass classification problems
	the ground truth is expected to be a 1D tensor with the corresponding
	classes. See Examples below

.. autoclass:: pytorch_widedeep.losses.MSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.MSLELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.RMSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.RMSLELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.QuantileLoss
	:members:

.. autoclass:: pytorch_widedeep.losses.FocalLoss
	:members:

.. autoclass:: pytorch_widedeep.losses.BayesianSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.TweedieLoss
	:members:

.. autoclass:: pytorch_widedeep.losses.ZILNLoss
	:members:

.. autoclass:: pytorch_widedeep.losses.L1Loss
	:members:

.. autoclass:: pytorch_widedeep.losses.FocalR_L1Loss
	:members:

.. autoclass:: pytorch_widedeep.losses.FocalR_MSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.FocalR_RMSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.HuberLoss
	:members:
