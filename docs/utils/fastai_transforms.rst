Fastai Transforms
=================
I have directly COPIED AND PASTED part of the ``transforms.py`` module from
the ``Fastai`` library. The reason to do such a thing is because
``pytorch_widedeep`` only needs the ``Tokenizer`` and the ``Vocab`` classes
there. This way I avoid the numerous fastai dependencies.

Credit for all the code in the ``fastai_transforms`` module to Jeremy Howard
and the fastai team. I only include the documentation here for completion, but
I strongly advise the user to read the ``Fastai`` `documentation
<https://www.pyimagesearch.com/>`_.

.. autosummary::
    :nosignatures:

    pytorch_widedeep.utils.fastai_transforms.SpacyTokenizer

.. autoclass:: pytorch_widedeep.utils.fastai_transforms.SpacyTokenizer
	:members:
	:undoc-members:
	:show-inheritance:


.. autosummary::
    :nosignatures:

    pytorch_widedeep.utils.fastai_transforms.Tokenizer

.. autoclass:: pytorch_widedeep.utils.fastai_transforms.Tokenizer
	:members:
	:undoc-members:
	:show-inheritance:

.. autosummary::
    :nosignatures:

    pytorch_widedeep.utils.fastai_transforms.Vocab

.. autoclass:: pytorch_widedeep.utils.fastai_transforms.Vocab
	:members:
	:undoc-members:
	:show-inheritance:
