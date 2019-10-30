##################################################
# Store Library Version
##################################################
import os.path

try:
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        __version__ = f.read().strip()
except Exception:
    raise


##################################################
# utils module accessible from pytorch-widedeep
##################################################
from .utils import dense_utils
from .utils import text_utils
from .utils import fastai_transforms
from .utils import image_utils