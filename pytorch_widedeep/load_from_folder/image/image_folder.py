import os
import os.path
from typing import Any, Tuple, Union, Callable, Optional

from PIL import Image

from pytorch_widedeep.wdtypes import Tensor

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


class BatchImagePreprocessor:
    def fit(self) -> "BatchImagePreprocessor":
        pass

    def transform(self) -> Tensor:
        pass

    def fit_transform(self) -> Tensor:
        pass


class ImageFolder:
    def __init__(
        self,
        directory: str,
        loader: Callable[[str], Any] = default_loader,
        image_processor: Optional[BatchImagePreprocessor] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.directory = directory
        self.loader = loader
        self.image_processor = image_processor
        self.extensions = extensions
        self.transforms = transforms

    def get_item(self, fname: str):
        assert has_file_allowed_extension(fname, self.extensions)

        path = os.path.join(self.directory, fname)
        sample = self.loader(path)
        if self.image_processor is not None:
            sample = self.image_processor.transform(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
