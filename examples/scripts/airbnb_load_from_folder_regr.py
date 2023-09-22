import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from pytorch_widedeep.models import TabResnet  # noqa: F401
from pytorch_widedeep.models import TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.training import TrainerFromFolder
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.preprocessing import (  # ChunkWidePreprocessor,
    ImagePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    ImageFromFolder,
    WideDeepDatasetFromFolder,
)

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    # The airbnb dataset, which you could get from here:
    # http://insideairbnb.com/get-the-data.html, is too big to be included in
    # our datasets module (when including images). Therefore, go there,
    # download it, and use the download_images.py script to get the images
    # and the airbnb_data_processing.py to process the data. We'll find
    # better datasets in the future ;). Note that here we are only using a
    # small sample to illustrate the use, so PLEASE ignore the results, just
    # focus on usage

    train_size = 800
    eval_size = 100
    test_size = 101
    chunksize = 100
    n_chunks = int(np.ceil(train_size / chunksize))

    path = "/Users/javierrodriguezzaurin/Projects/pytorch-widedeep/examples/tmp_data/airbnb/"
    train_fname = "airbnb_sample_train.csv"
    eval_fname = "airbnb_sample_eval.csv"
    test_fname = "airbnb_sample_test.csv"

    img_path = "/Users/javierrodriguezzaurin/Projects/pytorch-widedeep/examples/tmp_data/airbnb/property_picture/"
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
        n_chunks=n_chunks,
        default_embed_dim=8,
        verbose=0,
    )

    text_preprocessor = ChunkTextPreprocessor(
        n_chunks=n_chunks,
        text_col=text_col,
        n_cpus=1,
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_path,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([path, train_fname]), chunksize=chunksize)
    ):
        print(f"chunk in loop: {i}")
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    train_tab_folder = TabFromFolder(
        fname=train_fname,
        directory=path,
        target_col=target_col,
        preprocessor=tab_preprocessor,
        text_col=text_col,
        img_col=img_col,
    )
    eval_tab_folder = TabFromFolder(fname=eval_fname, reference=train_tab_folder)  # type: ignore[arg-type]
    test_tab_folder = TabFromFolder(
        fname=test_fname, reference=train_tab_folder, ignore_target=True  # type: ignore[arg-type]
    )

    text_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_folder = ImageFromFolder(preprocessor=img_preprocessor)

    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=train_size,
        tab_folder=train_tab_folder,
        text_folder=text_folder,
        img_folder=img_folder,
    )
    eval_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=eval_size, tab_folder=eval_tab_folder, reference=train_dataset_folder
    )
    test_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=test_size, tab_folder=test_tab_folder, reference=train_dataset_folder
    )
    train_loader = DataLoader(train_dataset_folder, batch_size=16, num_workers=1)
    eval_loader = DataLoader(eval_dataset_folder, batch_size=16, num_workers=1)
    test_loader = DataLoader(test_dataset_folder, batch_size=16, num_workers=1)

    basic_rnn = BasicRNN(
        vocab_size=len(text_preprocessor.vocab.itos),
        embed_dim=32,
        hidden_dim=64,
        n_layers=2,
        head_hidden_dims=[100, 50],
    )

    deepimage = Vision(
        pretrained_model_name="resnet18", n_trainable=0, head_hidden_dims=[200, 100]
    )

    deepdense = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=cont_cols,
    )

    model = WideDeep(
        deeptabular=deepdense,
        deeptext=basic_rnn,
        deepimage=deepimage,
    )

    callbacks = [EarlyStopping, ModelCheckpoint(filepath="model_weights/wd_out.pt")]

    trainer = TrainerFromFolder(
        model,
        objective="regression",
        callbacks=callbacks,
    )

    trainer.fit(
        train_loader=train_loader,
        eval_loader=eval_loader,
        finetune=True,
        finetune_epochs=1,
    )
    preds = trainer.predict(test_loader=test_loader)
