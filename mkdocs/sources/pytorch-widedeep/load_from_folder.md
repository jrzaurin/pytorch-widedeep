# The ``load_from_folder`` module

The ``load_from_folder`` module contains the classes that are necessary to
load data from disk and these are inspired by the
[`ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
class in the `torchvision` library. This module is designed with one specific case in mind.
Such case is the following: given a multi-modal dataset with tabular data,
images and text, the images do not fit in memory, and therefore, they have to
be loaded from disk. However, as any other functionality in this library,
there is some flexibility and some additional cases can also be addressed
using this module.

For this module to be used, the datasets must be prepared in a certain way:

1. the tabular data must contain a column with the images names as stored in
disk, including the extension (`.jpg`, `.png`, etc...).

2. Regarding to the text dataset, the tabular data can contain a column with
the texts themselves or the names of the files containing the texts as stored
in disk.

The tabular data might or might not fit in disk itself. If it does
not, please see the ``ChunkPreprocessor`` utilities at the[``preprocessing``]
(preprocessing.md) module and the examples folder in the repo, which
illustrate such case. Finally note that only `csv` format is currently
supported in that case(more formats coming soon).

::: pytorch_widedeep.load_from_folder.tabular.tabular_from_folder.TabFromFolder

::: pytorch_widedeep.load_from_folder.tabular.tabular_from_folder.WideFromFolder

::: pytorch_widedeep.load_from_folder.text.text_from_folder.TextFromFolder

::: pytorch_widedeep.load_from_folder.image.image_from_folder.ImageFromFolder

::: pytorch_widedeep.load_from_folder.wd_dataset_from_folder.WideDeepDatasetFromFolder
