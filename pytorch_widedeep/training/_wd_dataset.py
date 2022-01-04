import numpy as np
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.utils.deeptabular_utils import get_kernel_window, find_bin


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
    lds: bool
        option wether the Label Distribution Smoothing will be applied to dataset
    lds_kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian'
        choice of kernel for Label Distribution Smoothing
    lds_ks: int = 5
        LDS kernel window size
    lds_sigma: int = 2
        standard deviation of ['gaussian','laplace'] kernel for LDS
    lds_granularity: int = 100,
        number of bins in histogram used in LDS to count occurence of sample values
    lds_reweight: Literal["sqrt", None] = None
        option to reweight bin frequency counts in LDS
    lds_Ymax: Optional[float] = None
        option to restrict LDS bins by upper label limit
    lds_Ymin: Optional[float] = None
        option to restrict LDS bins by lower label limit
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
        lds_kernel: Literal["gaussian", "triang", "laplace", None] = "gaussian",
        lds_ks: int = 5,
        lds_sigma: Union[int, float] = 2,
        lds_granularity: int = 100,
        lds_reweight: Literal["sqrt", None] = None,
        lds_Ymax: Optional[float] = None,
        lds_Ymin: Optional[float] = None,
    ):
        super(WideDeepDataset, self).__init__()
        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        self.lds = lds
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []
        self.Y = target
        if self.lds:
            if self.Y is not None:
                if lds_Ymax is None:
                    self.lds_Ymax = max(self.Y)
                else:
                    self.lds_Ymax = lds_Ymax
                if lds_Ymin is None:
                    self.lds_Ymin = min(self.Y)
                else:
                    self.lds_Ymin = lds_Ymin
            self.weights = self._prepare_weights(
                granularity=lds_granularity,
                reweight=lds_reweight,
                kernel=lds_kernel,
                ks=lds_ks,
                sigma=lds_sigma,
            )

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
            if self.lds:
                weight = np.asarray([self.weights[idx]]).astype("float32")
                return X, y, weight
            else:
                return X, y
        else:
            return X

    def _prepare_weights(
        self,
        granularity: int = 100,
        reweight: Literal["sqrt", None] = None,
        kernel: Literal["gaussian", "triang", "laplace", None] = "gaussian",
        ks: int = 5,
        sigma: Union[int, float] = 2,
    ):
        """Assign weight to each sample by following procedure:
        1.      creating histogram from label values with nuber of bins = granularity
        2[opt]. reweighting label frequencies by sqrt
        3[opt]. smoothing label frequencies by convolution of kernel function window with frequencies list
        4.      inverting values by n_samples / (n_classes * np.bincount(y)), see:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
        5.      assigning weight to each sample from closest bin value


        Parameters
        ----------
        granularity: int = 100,
            number of bins in histogram used in LDS to count occurence of sample values
        reweight: Literal["sqrt", None] = None
            option to reweight bin frequency counts in LDS
        kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian'
            choice of kernel for Label Distribution Smoothing
        ks: int = 5
            LDS kernel window size
        sigma: int = 2
            standard deviation of ['gaussian','laplace'] kernel for LDS

        Returns
        -------
        weights: list
            list with weight for each sample
        """

        bin_edges = np.linspace(
            self.lds_Ymin, self.lds_Ymax, num=granularity, endpoint=True
        )
        labels = self.Y
        value_dict = dict(zip(bin_edges[:-1], np.histogram(labels, bin_edges)[0]))

        if reweight == "sqrt":
            value_dict = dict(
                zip(value_dict.keys(), np.sqrt(list(value_dict.values())))
            )

        if kernel:
            lds_kernel_window = get_kernel_window(kernel, ks, sigma)
            smoothed_values = convolve1d(
                list(value_dict.values()), weights=lds_kernel_window, mode="constant"
            )

            weigths = sum(smoothed_values) / (len(smoothed_values) * smoothed_values)
            value_dict = dict(zip(value_dict.keys(), weigths))
        else:
            values = list(value_dict.values())
            weigths = sum(values) / (len(values) * values)
            value_dict = dict(zip(value_dict.keys(), weigths))

        left_bin_edges = find_bin(bin_edges, labels)
        weights = [value_dict[edge] for edge in left_bin_edges]

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
