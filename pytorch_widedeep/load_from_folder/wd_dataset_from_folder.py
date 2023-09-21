from typing import Optional

from sklearn.utils import Bunch
from torch.utils.data import Dataset

from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    ImageFromFolder,
)


class WideDeepDatasetFromFolder(Dataset):
    def __init__(
        self,
        n_samples: int,
        tab_folder: Optional[TabFromFolder],
        wide_folder: Optional[TabFromFolder] = None,
        text_folder: Optional[TextFromFolder] = None,
        img_folder: Optional[ImageFromFolder] = None,
    ):
        super(WideDeepDatasetFromFolder, self).__init__()

        self.n_samples = n_samples
        self.tab_folder = tab_folder
        self.wide_folder = wide_folder
        self.text_folder = text_folder
        self.img_folder = img_folder

    def __getitem__(self, idx: int):  # noqa: C901
        x = Bunch()
        X_tab, text_fname_or_text, img_fname, y = self.tab_folder.get_item(idx=idx)
        x.deeptabular = X_tab

        if self.wide_folder is not None:
            X_wide, _, _, _ = self.wide_folder.get_item(idx=idx)
            x.wide = X_wide

        if text_fname_or_text is not None:
            X_text = self.text_folder.get_item(text_fname_or_text)
            x.deeptext = X_text

        if img_fname is not None:
            X_img = self.img_folder.get_item(img_fname)
            x.deepimage = X_img

        # We are aware that returning sometimes X and sometimes X, y is not
        # the best practice, but is the easiest way at this stage
        if y is not None:
            return x, y
        else:
            return x

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    from pytorch_widedeep.preprocessing import (  # ChunkWidePreprocessor,
        ImagePreprocessor,
        ChunkTabPreprocessor,
        ChunkTextPreprocessor,
    )

    fname = "airbnb_sample.csv"

    fname = "airbnb_sample.csv"
    img_col = "id"
    text_col = "description"
    target_col = "yield"
    cat_embed_cols = [
        "host_listings_count",
        "neighbourhood_cleansed",
        "is_location_exact",
        "property_type",
        "room_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "guests_included",
        "minimum_nights",
        "instant_bookable",
        "cancellation_policy",
        "has_house_rules",
        "host_gender",
        "accommodates_catg",
        "guests_included_catg",
        "minimum_nights_catg",
        "host_listings_count_catg",
        "bathrooms_catg",
        "bedrooms_catg",
        "beds_catg",
        "security_deposit",
        "extra_people",
    ]
    cont_cols = ["latitude", "longitude"]

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_embed_cols,
        continuous_cols=cont_cols,
        n_chunks=11,
        default_embed_dim=8,
        verbose=0,
    )

    text_preprocessor = ChunkTextPreprocessor(
        n_chunks=11,
        text_col=text_col,
        n_cpus=1,
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path="/Users/javierrodriguezzaurin/Projects/pytorch-widedeep/examples/tmp_data/airbnb/property_picture/",
    )

    chunksize = 100
    for i, chunk in enumerate(pd.read_csv(fname, chunksize=chunksize)):
        print(f"chunk in loop: {i}")
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    tab_folder = TabFromFolder(
        directory="",
        fname=fname,
        target_col=target_col,
        preprocessor=tab_preprocessor,
        text_col=text_col,
        img_col=img_col,
    )

    text_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_folder = ImageFromFolder(preprocessor=img_preprocessor)

    processed_sample, text_fname_or_text, img_fname, target = tab_folder.get_item(
        idx=10
    )

    text_sample = text_folder.get_item(text_fname_or_text)

    img_sample = img_folder.get_item(img_fname)

    dataset_folder = WideDeepDatasetFromFolder(
        n_samples=1001,
        tab_folder=tab_folder,
        text_folder=text_folder,
        img_folder=img_folder,
    )

    data_folder_loader = DataLoader(dataset_folder, batch_size=32, num_workers=1)

    for i, data in tqdm(enumerate(data_folder_loader), total=len(data_folder_loader)):
        X, y = data
