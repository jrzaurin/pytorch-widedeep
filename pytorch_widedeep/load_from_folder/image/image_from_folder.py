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
        directory: Optional[str] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transforms: Optional[Any] = None,
    ) -> None:
        assert (
            directory is not None or preprocessor is not None
        ), "Either a directory or an instance of ImagePreprocessor must be provided"

        if directory is not None and preprocessor is not None:  # pragma: no cover
            assert directory == preprocessor.img_path, (
                "If both 'directory' and 'preprocessor' are provided, the 'img_path' "
                "attribute of the 'preprocessor' must be the same as the 'directory'"
            )

        self.directory = directory if directory is not None else preprocessor.img_path
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

        self.img_path = directory if directory is not None else preprocessor.img_path

    def get_item(self, fname: str) -> np.ndarray:
        assert has_file_allowed_extension(fname, self.extensions)

        path = os.path.join(self.directory, fname)
        sample = self.loader(path)

        assert isinstance(sample, (Image.Image, np.ndarray)), (  # pragma: no cover
            "The loader must return an instance of PIL.Image or np.ndarray, "
            f"got {type(sample)} instead"
        )

        if self.preprocessor is not None:
            if not isinstance(sample, np.ndarray):
                processed_sample = self.preprocessor.transform_sample(
                    np.asarray(sample)
                )
            else:
                processed_sample = self.preprocessor.transform_sample(sample)
        else:
            processed_sample = sample

        prepared_sample = self._prepare_sample(processed_sample)

        return prepared_sample

    def _prepare_sample(self, processed_sample: Union[np.ndarray, Image.Image]):
        # if an image dataset is used, make sure is in the right format to
        # be ingested by the conv layers

        if isinstance(processed_sample, Image.Image):
            if not self.transforms:
                raise UserWarning(  # pragma: no cover
                    "The images are in PIL Image format, and not 'transforms' are passed. "
                    "This loader will simply return the array representation of the PIL Image. "
                )
                processed_sample = np.asarray(processed_sample)
            else:
                processed_sample = self.transforms(processed_sample)
        else:
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

            if not self.transforms or "ToTensor" not in self.transforms_names:
                # if there are no transforms, or these do not include ToTensor()
                # (weird or unexpected case, not sure is even possible) then we need
                # to  replicate what ToTensor() does -> transpose axis and normalize if
                # necessary
                if isinstance(processed_sample, Image.Image):
                    processed_sample = np.asarray(processed_sample)

                if processed_sample.ndim == 2:
                    processed_sample = processed_sample[:, :, None]

                processed_sample = processed_sample.transpose(2, 0, 1)

                if "int" in str(processed_sample.dtype):
                    processed_sample = (
                        processed_sample / processed_sample.max()
                    ).astype("float32")
            elif "ToTensor" in self.transforms_names:
                # if ToTensor() is included, simply apply transforms
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
        return f"TabFromFolder({all_params.format(**self.__dict__)})"
