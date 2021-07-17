Metrics
=======

.. note:: Metrics in this module expect the predictions and ground truth to have the
	same dimensions for regression and binary classification problems (i.e.
	:math:`N_{samples}, 1)`. In the case of multiclass classification problems the
	ground truth is expected to be a 1D tensor with the corresponding classes.
	See Examples below

We have added the possibility of using the metrics available at the
`torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ library.
Note that this library is still in its early versions and therefore this
option should be used with caution. To use ``torchmetrics`` simply import
them and use them as any of the ``pytorch-widedeep`` metrics described
below.

.. code-block:: python

	from torchmetrics import Accuracy, Precision

	accuracy = Accuracy(average=None, num_classes=2)
	precision = Precision(average='micro', num_classes=2)

	trainer = Trainer(model, objective="binary", metrics=[accuracy, precision])

A functioning example for ``pytorch-widedeep`` using ``torchmetrics`` can be
found in the `Examples folder <https://github.com/jrzaurin/pytorch-widedeep/blob/master/examples>`_.


.. autoclass:: pytorch_widedeep.metrics.Accuracy
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.metrics.Precision
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.metrics.Recall
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.metrics.FBetaScore
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.metrics.F1Score
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.metrics.R2Score
	:members:
	:undoc-members:
