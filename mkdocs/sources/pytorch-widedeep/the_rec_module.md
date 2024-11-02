# The ``rec`` module

This module contains models are that specifically designed for recommendation systems.
While the rest of the models can be accessed from the ``pytorch_widedeep.models`` module, models
in this module need to be specifically imported from the ``rec`` module, e.g.:

```python
from pytorch_widedeep.models.rec import DeepFactorizationMachine
```

The list of models here is not meant to be exhaustive, but it includes some
common architectures such as factorization machines, field aware
factorization machines or extreme factorization machines. More models will be
added in the future.

::: pytorch_widedeep.models.rec.deepfm.DeepFactorizationMachine
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.deepffm.DeepFieldAwareFactorizationMachine
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.din.DeepInterestNetwork
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.xdeepfm.ExtremeDeepFactorizationMachine
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.autoint.AutoInt
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.autoint_plus.AutoIntPlus
    selection:
        filters:
            - "!^forward$"


::: pytorch_widedeep.models.rec.basic_transformer.Transformer
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.dcn.DeepCrossNetwork
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.dcnv2.DeepCrossNetworkV2
    selection:
        filters:
            - "!^forward$"

::: pytorch_widedeep.models.rec.gdcn.GatedDeepCrossNetwork
    selection:
        filters:
            - "!^forward$"
