Fastai Transforms
=================

I have directly copied and pasted part of the ``transforms.py`` module from
the ``fastai`` library. The reason to do such a thing is because
``pytorch_widedeep`` only needs the ``Tokenizer`` and the ``Vocab`` classes
there. This way I avoid extra dependencies. Credit for all the code in the
``fastai_transforms`` module to Jeremy Howard and the `fastai` team. I only
include the documentation here for completion, but I strongly advise the user
to read the ``fastai`` `documentation <https://docs.fast.ai/>`_.

.. autoclass:: pytorch_widedeep.utils.fastai_transforms.Tokenizer
	:members:
	:undoc-members:

.. autoclass:: pytorch_widedeep.utils.fastai_transforms.Vocab
	:members:
	:undoc-members:
