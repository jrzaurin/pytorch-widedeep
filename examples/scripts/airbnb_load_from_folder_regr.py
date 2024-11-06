import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pytorch_widedeep.models import TabResnet  # noqa: F401
from pytorch_widedeep.models import TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.training import TrainerFromFolder
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.preprocessing import (
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

if __name__ == "__main__":
    # The airbnb dataset, which you could get from here:
    # http://insideairbnb.com/get-the-data.html, is too big to be included in
    # our datasets module (when including images). Therefore, go there,
    # download it, and use the download_images.py script to get the images
    # and the airbnb_data_processing.py to process the data. We'll find
    # better datasets in the future ;). Note that here we are only using a
    # small sample to illustrate the use, so PLEASE ignore the results, just
    # focus on usage

    # For this exercise, we use a small sample of the airbnb dataset,
    # comprised of tabular data with a text column ('description') and an
    # image column ('id') that point to the images of the properties listed
    # in Airbnb. We know the size of the sample before hand (1001) so we set
    # a series of parameters accordingly
    train_size = 800
    eval_size = 100
    test_size = 101
    chunksize = 100
    n_chunks = int(np.ceil(train_size / chunksize))

    data_path = "../tmp_data/airbnb/"
    train_fname = "airbnb_sample_train.csv"
    eval_fname = "airbnb_sample_eval.csv"
    test_fname = "airbnb_sample_test.csv"

    # the images are stored in the 'property_picture' while the text is a
    # column in the 'airbnb_sample' dataframe. Let's then define the dir and
    # file variables
    img_path = "../tmp_data/airbnb/property_picture/"
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

    # Now, processing the data from here on can be done in two ways:
    # 1. The tabular data itsel fits in memory as is only the images that do
    # not: in this case you could use the 'standard' Preprocessors and off
    # you go, move directly to the '[...]FromFolder' functionalities

    # 2. The tabular data is also very large and does not fit in memory, so we
    # have to process it in chuncks. For this second case I have created the
    # Chunk Processors (Wide, Tab and Text). Note that at the moment ONLY csv
    # format is allowed for the tabular file. More formats will be supported
    # in the future.

    # For the following I will assume (simply for illustration purposes) that
    # we are in the second case. Nonetheless, the process, whether 1 or 2,
    # can be summarised as follows:
    # 1. Process the data
    # 2. Define the loaders from folder
    # 3. Define the datasets and dataloaders
    # 4. Define the model and the Trainer
    # 5. Fit the model and Predict

    # Process the data in Chunks
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

    # Note that all the (pre)processing of the images will occur 'on the fly',
    # as they are loaded from disk. Therefore, the flow for the image dataset
    # and for the tabular and text data modes is not entirely the same.
    # Tabular and text data uses Chunk processors while such processing
    # approach is not needed for the images
    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_path,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_path, train_fname]), chunksize=chunksize)
    ):
        print(f"chunk in loop: {i}")
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    # Instantiate the loaders from folder: again here some explanation is
    # required. As I mentioned earlier the "[...]FromFolder" functionalities
    # are thought for the case when we have tabular and text and/or image
    # datasets and the latter do not fit in memory, so they have to be loaded
    # from disk. With this in mind, the tabular data is the reference, and
    # must have columns that point to the image files and to the text files
    # (in case these exists instead of a column with the texts). Since the
    # tabular data is used as a reference, is the one that has to be splitted
    # in train/validation/test. The test and image 'FromFolder' objects only
    # point to the corresponding column or files, and therefore, we do not
    # need to create a separate instance per train/validation/test dataset
    train_tab_folder = TabFromFolder(
        fname=train_fname,
        directory=data_path,
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

    # Following 'standard' pytorch approaches, we define the datasets and then
    # the dataloaders
    train_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=train_size,
        tab_from_folder=train_tab_folder,
        text_from_folder=text_folder,
        img_from_folder=img_folder,
    )
    eval_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=eval_size,
        tab_from_folder=eval_tab_folder,
        reference=train_dataset_folder,
    )
    test_dataset_folder = WideDeepDatasetFromFolder(
        n_samples=test_size,
        tab_from_folder=test_tab_folder,
        reference=train_dataset_folder,
    )
    train_loader = DataLoader(train_dataset_folder, batch_size=16, num_workers=1)
    eval_loader = DataLoader(eval_dataset_folder, batch_size=16, num_workers=1)
    test_loader = DataLoader(test_dataset_folder, batch_size=16, num_workers=1)

    # And from here on, is all pretty standard within the library
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
