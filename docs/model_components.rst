The ``models`` module
======================

This module contains the models that can be used as the four main components
that will comprise a Wide and Deep model (``wide``, ``deeptabular``,
``deeptext``, ``deepimage``), as well as the ``WideDeep`` "constructor"
class``. Note that each of the four components can be used independently.

.. autoclass:: pytorch_widedeep.models.tabular.linear.wide.Wide
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.mlp.tab_mlp.TabMlp
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.mlp.context_attention_mlp.ContextAttentionMLP
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.mlp.self_attention_mlp.SelfAttentionMLP
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.resnet.tab_resnet.TabResnet
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.tabnet.tab_net.TabNet
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.transformers.tab_transformer.TabTransformer
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.transformers.saint.SAINT
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.transformers.ft_transformer.FTTransformer
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.transformers.tab_perceiver.TabPerceiver
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.tabular.transformers.tab_fastformer.TabFastFormer
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.text.attentive_rnn.BasicRNN
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.text.attentive_rnn.AttentiveRNN
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.text.stacked_attentive_rnn.StackedAttentiveRNN
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.image.vision.Vision
    :exclude-members: forward
    :members:

.. autoclass:: pytorch_widedeep.models.wide_deep.WideDeep
    :exclude-members: forward
    :members:
