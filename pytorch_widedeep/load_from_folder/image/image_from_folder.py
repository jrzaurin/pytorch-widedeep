import os
import os.path
from typing import Any, Tuple, Union, Callable, Optional

import numpy as np
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


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFromFolder:
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

        if directory is not None and preprocessor is not None:
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

        if self.preprocessor is not None:
            if not isinstance(sample, np.ndarray):
                processed_sample = self.preprocessor.transform_sample(
                    np.asarray(sample)
                )
            else:
                processed_sample = self.preprocessor.transform_sample(sample)

        if self.transforms is not None:
            processed_sample = self.transforms(sample)

        if (
            self.transforms is not None and "ToTensor" not in self.transforms_names
        ) or self.transforms is None:
            # it is weird or unexpected to have transforms and not having
            # one that converts to Tensor, in fact I am not even sure is
            # possible
            processed_sample = processed_sample.transpose(2, 0, 1)

        return processed_sample
