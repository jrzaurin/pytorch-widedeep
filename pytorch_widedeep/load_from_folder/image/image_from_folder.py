import os
import os.path
from typing import Any, List, Tuple, Union, Callable, Optional

import numpy as np
import torch
from PIL import Image

from pytorch_widedeep.preprocessing.image_preprocessor import ImagePreprocessor

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def has_file_allowed_extension(
    filename: str, extensions: Union[str, Tuple[str, ...]]
) -> bool:
    return filename.lower().endswith(
        extensions if isinstance(extensions, str) else tuple(extensions)
    )


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TO DO: specify the return type
def accimage_loader(path: str) -> Any:  # pragma: no cover
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":  # pragma: no cover
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFromFolder:
    """
    This class is used to load the image dataset from disk. It is inspired by
    the `ImageFolder` class at the `torchvision` library. Here, we have
    simply adapted to work within the context of a Wide and Deep multi-modal
    model.

    For examples, please, see the examples folder in the repo.

    Parameters
    ----------
    directory: str, Optional, default = None
        the path to the directory where the images are located. If None, a
        preprocessor must be provided.
    preprocessor: `ImagePreprocessor`, Optional, default = None
        a fitted `ImagePreprocessor` object.
    loader: Callable[[str], Any], Optional, default = default_loader
        a function to load a sample given its path.
    extensions: Tuple[str, ...], Optional, default = IMG_EXTENSIONS
        a tuple with the allowed extensions. If None, IMG_EXTENSIONS will be
        used where IMG_EXTENSIONS
        =".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"
    transforms: Optional[Any], default = None
        a `torchvision.transforms` object. If None, this class will simply
        return an array representation of the PIL Image
    """

    def __init__(
        self,
        directory: Optional[Union[str, List[str]]] = None,
        preprocessor: Optional[
            Union[ImagePreprocessor, List[ImagePreprocessor]]
        ] = None,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transforms: Optional[Any] = None,
    ) -> None:
        assert (
            directory is not None or preprocessor is not None
        ), "Either a directory or an instance of ImagePreprocessor(s) must be provided"

        if directory is not None and preprocessor is not None:  # pragma: no cover
            error_msg = (
                "If both 'directory' and 'preprocessor' are provided, the 'img_path' "
                "attribute of the 'preprocessor' must be the same as the 'directory'"
            )
            if isinstance(directory, list):
                assert isinstance(preprocessor, list)
                assert len(directory) == len(preprocessor)
                for d, p in zip(directory, preprocessor):
                    assert d == p.img_path, error_msg
            else:
                assert isinstance(preprocessor, ImagePreprocessor)
                assert directory == preprocessor.img_path, error_msg

        if directory is not None:
            self.directory = directory
        else:
            assert (
                preprocessor is not None
            ), "Either a directory or an instance of ImagePreprocessor must be provided"
            if isinstance(preprocessor, list):
                self.directory = [p.img_path for p in preprocessor]
            else:
                self.directory = preprocessor.img_path

        self.preprocessor = preprocessor
        self.loader = loader
        self.extensions = extensions if extensions is not None else IMG_EXTENSIONS
        self.transforms = transforms
        if self.transforms:
            self.transforms_names = [
                tr.__class__.__name__ for tr in self.transforms.transforms
            ]
        else:
            self.transforms_names = []

            self.transpose = True

    def get_item(
        self, fname: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(fname, list):
            if not isinstance(self.directory, list):
                _directory = [self.directory] * len(fname)
            else:
                _directory = self.directory
            if self.preprocessor is not None:
                assert isinstance(self.preprocessor, list)
                processed_sample: Union[np.ndarray, List[np.ndarray]] = [
                    self._preprocess_one_sample(f, d, p)
                    for f, d, p in zip(fname, _directory, self.preprocessor)
                ]
            else:
                processed_sample = [
                    self._preprocess_one_sample(f, d) for f, d in zip(fname, _directory)
                ]
        else:
            assert isinstance(self.directory, str)
            if self.preprocessor is not None:
                assert isinstance(self.preprocessor, ImagePreprocessor)
                processed_sample = self._preprocess_one_sample(
                    fname, self.directory, self.preprocessor
                )
            else:
                processed_sample = self._preprocess_one_sample(fname, self.directory)

        return processed_sample

    def _preprocess_one_sample(
        self,
        fname: str,
        directory: str,
        preprocessor: Optional[ImagePreprocessor] = None,
    ) -> np.ndarray:
        assert has_file_allowed_extension(fname, self.extensions)

        path = os.path.join(directory, fname)
        sample = self.loader(path)

        assert isinstance(sample, (Image.Image, np.ndarray)), (  # pragma: no cover
            "The loader must return an instance of PIL.Image or np.ndarray, "
            f"got {type(sample)} instead"
        )

        if preprocessor is not None:
            if not isinstance(sample, np.ndarray):
                processed_sample = preprocessor.transform_sample(np.asarray(sample))
            else:
                processed_sample = preprocessor.transform_sample(sample)
        else:
            processed_sample = sample

        prepared_sample = self._prepare_sample(processed_sample)

        return prepared_sample

    def _prepare_sample(  # noqa: C901
        self, processed_sample: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        # if an image dataset is used, make sure is in the right format to
        # be ingested by the conv layers

        if isinstance(processed_sample, Image.Image):
            processed_sample = np.asarray(processed_sample)

        # if int must be uint8
        if "int" in str(processed_sample.dtype) and "uint8" != str(
            processed_sample.dtype
        ):
            processed_sample = processed_sample.astype("uint8")

        # if float must be float32
        if "float" in str(processed_sample.dtype) and "float32" != str(
            processed_sample.dtype
        ):
            processed_sample = processed_sample.astype("float32")

        # if there are no transforms, or these do not include ToTensor()
        # (weird or unexpected case, not sure is even possible) then we need
        # to  replicate what ToTensor() does -> transpose axis and normalize if
        # necessary
        if not self.transforms or "ToTensor" not in self.transforms_names:
            if processed_sample.ndim == 2:
                processed_sample = processed_sample[:, :, None]

            processed_sample = processed_sample.transpose(2, 0, 1)

            if "int" in str(processed_sample.dtype):
                processed_sample = (processed_sample / processed_sample.max()).astype(
                    "float32"
                )
        elif "ToTensor" in self.transforms_names:
            # if ToTensor() is included, simply apply transforms
            assert self.transforms_names[0] == "ToTensor", (
                "If ToTensor() is included in the transforms, it must be the "
                "first transform in the list"
            )
            processed_sample = self.transforms(processed_sample)
        else:
            # else apply transforms on the result of calling torch.tensor on
            # processed_sample after all the previous manipulation
            processed_sample = self.transforms(torch.tensor(processed_sample))

        return processed_sample

    def __repr__(self) -> str:
        list_of_params: List[str] = []
        if self.directory is not None:
            list_of_params.append("directory={directory}")
        if self.preprocessor is not None:
            list_of_params.append(
                f"preprocessor={self.preprocessor.__class__.__name__}"
            )
        if self.loader is not None:
            list_of_params.append(f"loader={self.loader.__name__}")
        if self.extensions is not None:
            list_of_params.append("extensions={extensions}")
        if self.transforms is not None:
            list_of_params.append(f"transforms={self.transforms_names}")
        all_params = ", ".join(list_of_params)
        return f"ImageFromFolder({all_params.format(**self.__dict__)})"
