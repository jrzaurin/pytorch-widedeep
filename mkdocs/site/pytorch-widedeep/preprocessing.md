# The ``preprocessing`` module

This module contains the classes that are used to prepare the data before
being passed to the models. There is one Preprocessor per data mode or
model component: ``wide``, ``deeptabular``, ``deepimage`` and ``deeptext``.

::: pytorch_widedeep.preprocessing.wide_preprocessor.WidePreprocessor

::: pytorch_widedeep.preprocessing.tab_preprocessor.TabPreprocessor

::: pytorch_widedeep.preprocessing.tab_preprocessor.Quantizer

::: pytorch_widedeep.preprocessing.text_preprocessor.TextPreprocessor

::: pytorch_widedeep.preprocessing.hf_preprocessor.HFPreprocessor

::: pytorch_widedeep.preprocessing.image_preprocessor.ImagePreprocessor


## Chunked versions

Chunked versions of the preprocessors are also available. These are useful
when the data is too big to fit in memory. See also the [``load_from_folder``](load_from_folder.md)
module in the library and the corresponding section here in the documentation.

Note that there is not a ``ChunkImagePreprocessor``. This is because the
processing of the images will occur inside the `ImageFromFolder` class in
the [``load_from_folder``](load_from_folder.md) module.


::: pytorch_widedeep.preprocessing.wide_preprocessor.ChunkWidePreprocessor

::: pytorch_widedeep.preprocessing.tab_preprocessor.ChunkTabPreprocessor

::: pytorch_widedeep.preprocessing.text_preprocessor.ChunkTextPreprocessor
