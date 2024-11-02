# Fastai transforms

We directly copied and pasted part of the ``transforms.py`` module from
the ``fastai`` library (from an old version). The reason to do such a thing is because
``pytorch_widedeep`` only needs the ``Tokenizer`` and the ``Vocab`` classes
there. This way we avoid extra dependencies. Credit for all the code in the
``fastai_transforms`` module in this ``pytorch-widedeep`` package goes to
Jeremy Howard and the `fastai` team. I only include the documentation here for
completion, but I strongly advise the user to read the ``fastai`` [documentation](https://docs.fast.ai/).

::: pytorch_widedeep.utils.fastai_transforms.Tokenizer

::: pytorch_widedeep.utils.fastai_transforms.Vocab
