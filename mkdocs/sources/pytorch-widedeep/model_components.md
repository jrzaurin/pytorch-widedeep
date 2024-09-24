# The ``models`` module

This module contains the models that can be used as the four main components
that will comprise a Wide and Deep model (``wide``, ``deeptabular``,
``deeptext``, ``deepimage``), as well as the ``WideDeep`` "constructor"
class. Note that each of the four components can be used independently. It
also contains all the documentation for the models that can be used for
self-supervised pre-training with tabular data.


::: pytorch_widedeep.models.tabular.linear.wide.Wide
    selection:
        filters:
            - "!^forward$"
            - "!^_reset_parameters$"

::: pytorch_widedeep.models.tabular.mlp.tab_mlp.TabMlp
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.mlp.tab_mlp.TabMlpDecoder
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.resnet.tab_resnet.TabResnet
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.resnet.tab_resnet.TabResnetDecoder
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.tabnet.tab_net.TabNet
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.tabnet.tab_net.TabNetDecoder
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.mlp.context_attention_mlp.ContextAttentionMLP
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.mlp.self_attention_mlp.SelfAttentionMLP
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.transformers.tab_transformer.TabTransformer
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.transformers.saint.SAINT
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.transformers.ft_transformer.FTTransformer
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.transformers.tab_perceiver.TabPerceiver
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.tabular.transformers.tab_fastformer.TabFastFormer
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.text.rnns.basic_rnn.BasicRNN
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.models.text.rnns.attentive_rnn.AttentiveRNN
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.models.text.rnns.stacked_attentive_rnn.StackedAttentiveRNN
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.models.text.huggingface_transformers.hf_model.HFModel
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"


::: pytorch_widedeep.models.image.vision.Vision
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.models.model_fusion.ModelFuser
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"

::: pytorch_widedeep.models.wide_deep.WideDeep
    selection:
        filters:
            - "!^_"  # exclude all members starting with _
            - "!^forward$"
