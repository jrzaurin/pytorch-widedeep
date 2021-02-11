Metrics
=======

.. note:: Metrics in this module expect the predictions and ground truth to have the
	same dimensions for regression and binary classification problems (i.e.
	:math:`N_{samples}, 1)`. In the case of multiclass classification problems the
	ground truth is expected to be a 1D tensor with the corresponding classes.
	See Examples below

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
