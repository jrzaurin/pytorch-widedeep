import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_widedeep.utils.image_utils import (
    SimplePreprocessor,
    AspectAwarePreprocessor,
)
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class ImagePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the ``deepimage`` input dataset.

    The Preprocessing consists simply on resizing according to their
    aspect ratio

    Parameters
    ----------
    img_col: str
        name of the column with the images filenames
    img_path: str
        path to the dicrectory where the images are stored
    width: int, default=224
        width of the resulting processed image.
    height: int, default=224
        width of the resulting processed image.
    verbose: int, default 1
        Enable verbose output.

    Attributes
    ----------
    aap: AspectAwarePreprocessor
        an instance of :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
    spp: SimplePreprocessor
        an instance of :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`
    normalise_metrics: Dict
        Dict containing the normalisation metrics of the image dataset, i.e.
        mean and std for the R, G and B channels

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> from pytorch_widedeep.preprocessing import ImagePreprocessor
    >>>
    >>> path_to_image1 = 'tests/test_data_utils/images/galaxy1.png'
    >>> path_to_image2 = 'tests/test_data_utils/images/galaxy2.png'
    >>>
    >>> df_train = pd.DataFrame({'images_column': [path_to_image1]})
    >>> df_test = pd.DataFrame({'images_column': [path_to_image2]})
    >>> img_preprocessor = ImagePreprocessor(img_col='images_column', img_path='.', verbose=0)
    >>> resized_images = img_preprocessor.fit_transform(df_train)
    >>> new_resized_images = img_preprocessor.transform(df_train)

    .. note:: Normalising metrics will only be computed when the
        ``fit_transform`` method is run. Running ``transform`` only will not
        change the computed metrics and running ``fit`` only simply
        instantiates the resizing functions.
    """

    def __init__(
        self,
        img_col: str,
        img_path: str,
        width: int = 224,
        height: int = 224,
        verbose: int = 1,
    ):
        super(ImagePreprocessor, self).__init__()

        self.img_col = img_col
        self.img_path = img_path
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        r"""Instantiates the Preprocessors
        :obj:`AspectAwarePreprocessor`` and :obj:`SimplePreprocessor` for image
        resizing.

        See
        :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
        and :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`.

        """
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)
        self._compute_normalising_metrics = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Resizes the images to the input height and width."""
        check_is_fitted(self, attributes=["aap"])
        image_list = df[self.img_col].tolist()
        if self.verbose:
            print("Reading Images from {}".format(self.img_path))
        imgs = [cv2.imread("/".join([self.img_path, img])) for img in image_list]

        # finding images with different height and width
        aspect = [(im.shape[0], im.shape[1]) for im in imgs]
        aspect_r = [a[0] / a[1] for a in aspect]
        diff_idx = [i for i, r in enumerate(aspect_r) if r != 1.0]

        if self.verbose:
            print("Resizing")
        resized_imgs = []
        for i, img in tqdm(enumerate(imgs), total=len(imgs), disable=self.verbose != 1):
            if i in diff_idx:
                resized_imgs.append(self.aap.preprocess(img))
            else:
                # if aspect ratio is 1:1, no need for AspectAwarePreprocessor
                resized_imgs.append(self.spp.preprocess(img))

        if self._compute_normalising_metrics:
            if self.verbose:
                print("Computing normalisation metrics")
            # mean and std deviation will only be computed when the fit method
            # is called
            mean_R, mean_G, mean_B = [], [], []
            std_R, std_G, std_B = [], [], []
            for rsz_img in resized_imgs:
                (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(
                    rsz_img
                )
                mean_R.append(mean_r)
                mean_G.append(mean_g)
                mean_B.append(mean_b)
                std_R.append(std_r)
                std_G.append(std_g)
                std_B.append(std_b)
            self.normalise_metrics = dict(
                mean={
                    "R": np.mean(mean_R) / 255.0,
                    "G": np.mean(mean_G) / 255.0,
                    "B": np.mean(mean_B) / 255.0,
                },
                std={
                    "R": np.mean(std_R) / 255.0,
                    "G": np.mean(std_G) / 255.0,
                    "B": np.mean(std_B) / 255.0,
                },
            )
            self._compute_normalising_metrics = False
        return np.asarray(resized_imgs)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def inverse_transform(self, transformed_image):
        raise NotImplementedError(
            "'inverse_transform' method is not implemented for 'ImagePreprocessor'"
        )
