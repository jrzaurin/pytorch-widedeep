Losses
======

``pytorch-widedeep`` accepts a number of losses and objectives that can be
passed to the ``Trainer`` class via the ``str`` parameter ``objective`` (see).
For most cases the loss function that ``pytorch-widedeep`` will use internally
is already implemented in Pytorch.

In addition, ``pytorch-widedeep`` implements four "custom" loss functions.
These are described below for completions since, as I mentioned before, they
are used internally by the ``Trainer``. Of course, onen could always use them
on their own and can be imported as:

.. note:: Losses in this module expect the predictions and ground truth to have the
	same dimensions for regression and binary classification problems (i.e. N
	samples, 1). In the case of multiclass classification problems the ground
	truth is expected to be a 1D tensor with the corresponding classes. See
	Examples below

.. code-block:: python

	from pytorch_widedeep.losses import FocalLoss

.. autoclass:: pytorch_widedeep.losses.FocalLoss
	:members:

.. autoclass:: pytorch_widedeep.losses.MSLELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.RMSELoss
	:members:

.. autoclass:: pytorch_widedeep.losses.RMSLELoss
	:members:
