import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from pytorch_widedeep.wdtypes import *  # noqa: F403

# TODO assert to limit the usage of LDS only for single value regression objective

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


class WideDeepDataset(Dataset):
    r"""
    Defines the Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray
        wide input
    X_tab: np.ndarray
        deeptabular input
    X_text: np.ndarray
        deeptext input
    X_img: np.ndarray
        deepimage input
    target: np.ndarray
        target array
    transforms: :obj:`MultipleTransforms`
        torchvision Compose object. See models/_multiple_transforms.py
    """

    def __init__(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_img: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Any] = None,
        lds: bool = False,
        lds_kernel: str = "gaussian",
        lds_ks: int = 5,
        lds_sigma: int = 2,
        reweight: str = None,
        Ymax: Optional[float] = None,
    ):
        super(WideDeepDataset, self).__init__()
        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        self.reweight = reweight
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []
        self.Y = target
        if self.Y is not None:
            if Ymax is None:
                self.Ymax = int(max(target))
            else:
                assert type(Ymax) == int, "Ymax values must be integer"
                self.Ymax = Ymax
        assert reweight in {None, "inverse", "sqrt_inv"}
        assert reweight != None if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
        if self.reweight != None:
            self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __getitem__(self, idx: int):  # noqa: C901
        X = Bunch()
        if self.X_wide is not None:
            X.wide = self.X_wide[idx]
        if self.X_tab is not None:
            X.deeptabular = self.X_tab[idx]
        if self.X_text is not None:
            X.deeptext = self.X_text[idx]
        if self.X_img is not None:
            # if an image dataset is used, make sure is in the right format to
            # be ingested by the conv layers
            xdi = self.X_img[idx]
            # if int must be uint8
            if "int" in str(xdi.dtype) and "uint8" != str(xdi.dtype):
                xdi = xdi.astype("uint8")
            # if int float must be float32
            if "float" in str(xdi.dtype) and "float32" != str(xdi.dtype):
                xdi = xdi.astype("float32")
            # if there are no transforms, or these do not include ToTensor(),
            # then we need to  replicate what Tensor() does -> transpose axis
            # and normalize if necessary
            if not self.transforms or "ToTensor" not in self.transforms_names:
                if xdi.ndim == 2:
                    xdi = xdi[:, :, None]
                xdi = xdi.transpose(2, 0, 1)
                if "int" in str(xdi.dtype):
                    xdi = (xdi / xdi.max()).astype("float32")
            # if ToTensor() is included, simply apply transforms
            if "ToTensor" in self.transforms_names:
                xdi = self.transforms(xdi)
            # else apply transforms on the result of calling torch.tensor on
            # xdi after all the previous manipulation
            elif self.transforms:
                xdi = self.transforms(torch.tensor(xdi))
            # fill the Bunch
            X.deepimage = xdi
        if self.Y is not None:
            y = self.Y[idx]
            if self.reweight != None:
                weight = np.asarray([self.weights[idx]]).astype("float32")
                return X, y, weight
            else:
                return X, y
        else:
            return X

    def _prepare_weights(self, reweight, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        max_target = self.Ymax
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.Y
        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == "sqrt_inv":
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == "inverse":
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == "none":
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f"Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})")
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode="constant")
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def __len__(self):
        if self.X_wide is not None:
            return len(self.X_wide)
        if self.X_tab is not None:
            return len(self.X_tab)
        if self.X_text is not None:
            return len(self.X_text)
        if self.X_img is not None:
            return len(self.X_img)
