from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from ..text.text_folder import TextFolder
from ..image.image_folder import ImageFolder
from ..tabular.tabular_folder import TabFolder


class WideDeepDatasetFolder(Dataset):
    def __init__(
        self,
        n_samples: int,
        filename: str,
        tab_folder: Optional[TabFolder],
        wide_folder: Optional[TabFolder] = None,
        text_folder: Optional[TextFolder] = None,
        img_folder: Optional[ImageFolder] = None,
    ):
        super(WideDeepDatasetFolder, self).__init__()

        self.n_samples = n_samples
        self.filename = filename
        self.tab_folder = tab_folder
        self.wide_folder = wide_folder
        self.text_folder = text_folder
        self.img_folder = img_folder

    def __getitem__(self, idx: int):  # noqa: C901
        X: Dict[str, Any] = {}
        X_tab, text_fname_or_text, img_fname, y = self.tab_folder.get_item(
            fname=self.filename, idx=idx
        )
        X["deeptabular"] = X_tab

        if self.wide_folder is not None:
            X_wide, _, _, _ = self.wide_folder.get_item(self.filename, idx=idx)
            X["wide"] = X_wide

        if text_fname_or_text is not None:
            X_text = self.text_folder.get_item(text_fname_or_text)
            X["deeptext"] = X_text

        if img_fname is not None:
            X_img = self.img_folder.get_item(img_fname)
            X["deepimage"] = X_img

        # We are aware that returning sometimes X and sometimes X, y is not
        # the best practice, but is th easiest way at this stage
        if y is not None:
            return X, y
        else:
            return X

    def __len__(self):
        self.n_samples
