# The ``bayesian models`` module

This module contains the two Bayesian Models available in this library, namely
the bayesian version of the ``Wide`` and ``TabMlp`` models, referred as
``BayesianWide`` and ``BayesianTabMlp``. These models are very useful in
scenarios where getting a measure of uncertainty is important.

The models in this module are based on the publication:
[Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424?context=cs).


::: pytorch_widedeep.bayesian_models.tabular.bayesian_linear.bayesian_wide.BayesianWide
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.bayesian_models.tabular.bayesian_mlp.bayesian_tab_mlp.BayesianTabMlp
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"
