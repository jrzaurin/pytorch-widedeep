from typing import Any, List, Type, Tuple, Optional

from sklearn.utils import Bunch
from torch.utils.data import Dataset

from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    WideFromFolder,
    ImageFromFolder,
)


class WideDeepDatasetFromFolder(Dataset):
    """
    This class is the Dataset counterpart of the `WideDeepDataset` class.

    Given a reference tabular dataset, with columns that indicate the path to
    the images and to the text files or the texts themselves, it will use the
    `[...]FromFolder` classes to load the data consistently from disk per batch.

    For examples, please, see the examples folder in the repo.

    Parameters
    ----------
    n_samples: int
        Number of samples in the dataset
    tab_from_folder: TabFromFolder
        Instance of the `TabFromFolder` class
    wide_from_folder: Optional[WideFromFolder], default = None
        Instance of the `WideFromFolder` class
    text_from_folder: Optional[TextFromFolder], default = None
        Instance of the `TextFromFolder` class
    img_from_folder: Optional[ImageFromFolder], default = None
        Instance of the `ImageFromFolder` class
    reference: Type["WideDeepDatasetFromFolder"], default = None
        If not None, the 'text_from_folder' and 'img_from_folder' objects will
        be retrieved from the reference class. This is useful when we want to
        use a `WideDeepDatasetFromFolder` class used for a train dataset as a
        reference for the validation and test datasets. In this case, the
        `text_from_folder` and `img_from_folder` objects will be the same for
        all three datasets, so there is no need to create a new instance for
        each dataset.
    """

    def __init__(
        self,
        n_samples: int,
        tab_from_folder: Optional[TabFromFolder] = None,
        wide_from_folder: Optional[WideFromFolder] = None,
        text_from_folder: Optional[TextFromFolder] = None,
        img_from_folder: Optional[ImageFromFolder] = None,
        reference: Optional[Any] = None,  # is Type["WideDeepDatasetFromFolder"],
    ):
        super(WideDeepDatasetFromFolder, self).__init__()

        if tab_from_folder is None and wide_from_folder is None:
            raise ValueError(
                "Either 'tab_from_folder' or 'wide_from_folder' must be not None"
            )

        if reference is not None:
            assert (
                img_from_folder is None and text_from_folder is None
            ), "If reference is not None, 'img_from_folder' and 'text_from_folder' left as None"
            self.text_from_folder, self.img_from_folder = self._get_from_reference(
                reference
            )
        else:
            assert (
                text_from_folder is not None and img_from_folder is not None
            ), "If reference is None, 'img_from_folder' and 'text_from_folder' must be not None"
            self.text_from_folder = text_from_folder
            self.img_from_folder = img_from_folder

        self.n_samples = n_samples
        self.tab_from_folder = tab_from_folder
        self.wide_from_folder = wide_from_folder

    def __getitem__(self, idx: int):  # noqa: C901
        x = (
            Bunch()
        )  # for consistency with WideDeepDataset, but this is just a Dict[str, Any]

        if self.tab_from_folder is not None:
            X_tab, text_fname_or_text, img_fname, y = self.tab_from_folder.get_item(
                idx=idx
            )
            x.deeptabular = X_tab

        if self.wide_from_folder is not None:
            if self.tab_from_folder is None:
                (
                    X_wide,
                    text_fname_or_text,
                    img_fname,
                    y,
                ) = self.wide_from_folder.get_item(idx=idx)
            else:
                X_wide, _, _, _ = self.wide_from_folder.get_item(idx=idx)
            x.wide = X_wide

        if text_fname_or_text is not None:
            # These assertions should never be raised, but just in case...
            assert (
                self.text_from_folder is not None
            ), "text_fname_or_text is not None but self.text_from_folder is None"
            X_text = self.text_from_folder.get_item(text_fname_or_text)
            x.deeptext = X_text

        if img_fname is not None:
            assert (
                self.img_from_folder is not None
            ), "img_fname is not None but self.img_from_folder is None"
            X_img = self.img_from_folder.get_item(img_fname)
            x.deepimage = X_img

        # We are aware that returning sometimes X and sometimes X, y is not
        # the best practice, but is the easiest way at this stage
        if y is not None:
            return x, y
        else:
            return x

    def __len__(self):
        return self.n_samples

    @staticmethod
    def _get_from_reference(
        reference: Type["WideDeepDatasetFromFolder"],
    ) -> Tuple[Optional[TextFromFolder], Optional[ImageFromFolder]]:
        return reference.text_from_folder, reference.img_from_folder

    def __repr__(self) -> str:
        list_of_params: List[str] = []
        list_of_params.append("n_samples={n_samples}")
        if self.tab_from_folder is not None:
            list_of_params.append(
                f"tab_from_folder={self.tab_from_folder.__class__.__name__}"
            )
        if self.wide_from_folder is not None:
            list_of_params.append(
                f"wide_from_folder={self.wide_from_folder.__class__.__name__}"
            )
        if self.text_from_folder is not None:
            list_of_params.append(
                f"text_from_folder={self.text_from_folder.__class__.__name__}"
            )
        if self.img_from_folder is not None:
            list_of_params.append(
                f"img_from_folder={self.img_from_folder.__class__.__name__}"
            )
        all_params = ", ".join(list_of_params)
        return f"WideDeepDatasetFromFolder({all_params.format(**self.__dict__)})"
