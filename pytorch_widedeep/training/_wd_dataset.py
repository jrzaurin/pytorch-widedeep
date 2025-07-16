from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from pytorch_widedeep.wdtypes import Optional, Transforms


class WideDeepDataset(Dataset):
    r"""
    Defines the Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray
        wide input
    X_tab: np.ndarray or List[np.ndarray]
        deeptabular input
    X_text: np.ndarray or List[np.ndarray]
        deeptext input
    X_img: np.ndarray or List[np.ndarray]
        deepimage input
    target: np.ndarray
        target array
    transforms: Optional[Transforms | Compose]
        torchvision Compose object. See models/_multiple_transforms.py
    """

    def __init__(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Union[Transforms, Compose]] = None,
    ):
        super(WideDeepDataset, self).__init__()
        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        if self.transforms:
            if isinstance(self.transforms, Compose):
                self.transforms_names = [
                    tr.__class__.__name__ for tr in self.transforms.transforms
                ]
            else:
                self.transforms_names = [self.transforms.__class__.__name__]
        else:
            self.transforms_names = []
        self.Y = target

    def __getitem__(self, idx: int):  # noqa: C901
        x = Bunch()
        if self.X_wide is not None:
            x.wide = self.X_wide[idx]
        if self.X_tab is not None:
            if isinstance(self.X_tab, list):
                x.deeptabular = [self.X_tab[i][idx] for i in range(len(self.X_tab))]
            else:
                x.deeptabular = self.X_tab[idx]
        if self.X_text is not None:
            if isinstance(self.X_text, list):
                x.deeptext = [self.X_text[i][idx] for i in range(len(self.X_text))]
            else:
                x.deeptext = self.X_text[idx]
        if self.X_img is not None:
            if isinstance(self.X_img, list):
                x.deepimage = [
                    self._prepare_images(self.X_img[i][idx])
                    for i in range(len(self.X_img))
                ]
            else:
                x.deepimage = self._prepare_images(self.X_img[idx])
        if self.Y is None:
            return x
        else:
            y = self.Y[idx]
            return x, y

    def _prepare_images(self, xdi: np.ndarray):
        # if an image dataset is used, make sure is in the right format to
        # be ingested by the conv layers
        # if int must be uint8
        if "int" in str(xdi.dtype) and "uint8" != str(xdi.dtype):
            xdi = xdi.astype("uint8")
        # if float must be float32
        if "float" in str(xdi.dtype) and "float32" != str(xdi.dtype):
            xdi = xdi.astype("float32")
        # if there are no transforms, or these do not include ToTensor(),
        # then we need to  replicate what Tensor() does -> transpose axis
        # and normalize if necessary
        if not self.transforms or "ToTensor" not in self.transforms_names:
            if xdi.ndim == 2:
                xdi = xdi[:, :, None]
            # transpose only if the 1st dimension is not channels (3)
            if xdi.shape[0] != 3:
                xdi = xdi.transpose(2, 0, 1)
            # normalize if necessary (heuristic: if int, then normalize)
            if "int" in str(xdi.dtype):
                xdi = (xdi / xdi.max()).astype("float32")
        # if ToTensor() is included, simply apply transforms
        if "ToTensor" in self.transforms_names:
            xdi = self.transforms(xdi)
        # else apply transforms on the result of calling torch.tensor on
        # xdi after all the previous manipulation
        elif self.transforms:
            xdi = self.transforms(torch.tensor(xdi))
        return xdi

    def __len__(self):
        if self.X_wide is not None:
            return len(self.X_wide)
        if self.X_tab is not None:
            if isinstance(self.X_tab, list):
                return len(self.X_tab[0])
            else:
                return len(self.X_tab)
        if self.X_text is not None:
            if isinstance(self.X_text, list):
                return len(self.X_text[0])
            else:
                return len(self.X_text)
        if self.X_img is not None:
            if isinstance(self.X_img, list):
                return len(self.X_img[0])
            else:
                return len(self.X_img)


# This code is currently not used in the library. It will eventually be used
# if I decide to support datasets for recommendation system with a varying
# number of items per user. Allowing this in the current implementation
# brings a series of issues that as of right now I prefer not to address.
class GroupWideDeepDataset(Dataset):
    def __init__(
        self,
        group_sizes: np.ndarray,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Union[Transforms, Compose]] = None,
    ):
        super(GroupWideDeepDataset, self).__init__()
        self.group_sizes = group_sizes
        self.group_cumsum = np.cumsum([0] + self.group_sizes)
        self.group_cumsum = np.insert(self.group_cumsum, 0, 0)
        self.num_groups = len(self.group_sizes)

        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img

        self.transforms = transforms
        if self.transforms:
            if isinstance(self.transforms, Compose):
                self.transforms_names = [
                    tr.__class__.__name__ for tr in self.transforms.transforms
                ]
            else:
                self.transforms_names = [self.transforms.__class__.__name__]
        else:
            self.transforms_names = []

        self.Y = target

    def __getitem__(self, idx: int):
        print(idx)
        start_idx = self.group_cumsum[idx]
        end_idx = self.group_cumsum[idx + 1]

        x = Bunch()
        if self.X_wide is not None:
            x.wide = self.X_wide[start_idx:end_idx]
        if self.X_tab is not None:
            if isinstance(self.X_tab, list):
                x.deeptabular = [
                    self.X_tab[i][start_idx:end_idx] for i in range(len(self.X_tab))
                ]
            else:
                x.deeptabular = self.X_tab[start_idx:end_idx]
        if self.X_text is not None:
            if isinstance(self.X_text, list):
                x.deeptext = [
                    self.X_text[i][start_idx:end_idx] for i in range(len(self.X_text))
                ]
            else:
                x.deeptext = self.X_text[start_idx:end_idx]
        if self.X_img is not None:
            if isinstance(self.X_img, list):
                x.deepimage = [
                    self._prepare_images(self.X_img[i][start_idx:end_idx])
                    for i in range(len(self.X_img))
                ]
            else:
                x.deepimage = self._prepare_images(self.X_img[start_idx:end_idx])

        if self.Y is None:
            return x
        else:
            y = self.Y[start_idx:end_idx]
            return x, y

    def _prepare_images(self, xdi: np.ndarray):
        if "int" in str(xdi.dtype) and "uint8" != str(xdi.dtype):
            xdi = xdi.astype("uint8")
        if "float" in str(xdi.dtype) and "float32" != str(xdi.dtype):
            xdi = xdi.astype("float32")
        if not self.transforms or "ToTensor" not in self.transforms_names:
            # first dim will be the size of the group, so ndim == 3 implies
            # 2D images
            if xdi.ndim == 3:
                xdi = xdi[:, :, :, None]
            # Group Size, Height, Width, Channel -> Group Size, Channel,
            # Height, Width
            if xdi.shape[1] != 3:
                xdi = xdi.transpose(0, 3, 1, 2)
            if "int" in str(xdi.dtype):
                xdi = (xdi / xdi.max()).astype("float32")
        # if ToTensor() is included, simply apply transforms
        if "ToTensor" in self.transforms_names:
            xdi = self.transforms(xdi)
        # else apply transforms on the result of calling torch.tensor on
        # xdi after all the previous manipulation
        elif self.transforms:
            xdi = self.transforms(torch.tensor(xdi))
        return xdi

    def __len__(self):
        return self.num_groups


def group_collate_fn(batch: List[Tuple[Bunch, np.ndarray]]):

    all_targets: List[np.ndarray] = []
    all_features: Dict[str, Union[List[np.ndarray], List[List[np.ndarray]]]] = {
        "wide": [],
        "deeptabular": [],
        "deeptext": [],
        "deepimage": [],
    }
    for features, targets in batch:
        for key in all_features.keys():
            if key in features:
                all_features[key].append(features[key])
        all_targets.append(targets)
    all_features = {k: v for k, v in all_features.items() if v}

    batch_features: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}

    for key, value in all_features.items():
        if isinstance(value[0], list):
            value = [
                np.concatenate([v[i] for v in value], axis=0)
                for i in range(len(value[0]))
            ]
            batch_features[key] = [torch.from_numpy(v) for v in value]
        else:
            batch_features[key] = torch.from_numpy(np.concatenate(value, axis=0))

    batch_targets = torch.from_numpy(np.concatenate(all_targets, axis=0))

    return batch_features, batch_targets
