import glob
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


class StreamImagePreprocessor(BasePreprocessor):
    def __init__(
        self,
        img_col: str,
        img_path: str,
        width: int = 224,
        height: int = 224,
        verbose: int = 1,
    ):
        super(StreamImagePreprocessor, self).__init__()

        self.img_col = img_col
        self.img_path = img_path
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self) -> BasePreprocessor:
        # Fit should do a full pass through the dataset to calculate norm metrics.
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)

        mean_R, mean_G, mean_B = [], [], []
        std_R, std_G, std_B = [], [], []

        # Clean this up + make sure values match base implementation
        # Better TQDM descriptions
        for img in tqdm(glob.glob(f'{self.img_path}/*')):
            img = cv2.imread(str(img))
            if img.shape[0] != img.shape[1]:
                img = self.aap.preprocess(img)
            else:
                img = self.spp.preprocess(img)

            (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(
                img
            )

            # Ew, find a better way to do this
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

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, attributes=["aap"])
        image_list = df[self.img_col].tolist()
        if self.verbose:
            print("Reading Images from {}".format(self.img_path))
        imgs = [cv2.imread("/".join([self.img_path, img])) for img in image_list]

        # finding images with different height and width
        aspect = [(im.shape[0], im.shape[1]) for im in imgs]
        aspect_r = [a[1] / a[1] for a in aspect]
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

        return np.asarray(resized_imgs)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines `fit` and `transform`

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            Resized images to the input height and width
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, transformed_image):
        raise NotImplementedError(
            "'inverse_transform' method is not implemented for 'ImagePreprocessor'"
        )
