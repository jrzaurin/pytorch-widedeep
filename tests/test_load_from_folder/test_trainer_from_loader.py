# The tests in this script are perhaps not the best I have written...I am
# literally doing this in bursts of a few minutes...
import os

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    Vision,
    BasicRNN,
    WideDeep,
    TabTransformer,
)
from pytorch_widedeep.training import TrainerFromFolder
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.preprocessing import (
    ImagePreprocessor,
    ChunkTabPreprocessor,
    ChunkTextPreprocessor,
    ChunkWidePreprocessor,
)
from pytorch_widedeep.load_from_folder import (
    TabFromFolder,
    TextFromFolder,
    WideFromFolder,
    ImageFromFolder,
    WideDeepDatasetFromFolder,
)

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

data_folder = os.path.join(path, "load_from_folder_test_data")
img_folder = os.path.join(data_folder, "images")

fname = "synthetic_dataset.csv"
img_col = "images"
text_col = "text"
cat_cols = ["category1", "category2"]
num_cols = ["numeric1", "numeric2"]

data_size = 32
chunksize = 8
n_chunks = data_size // chunksize


def _build_preprocessors(tab_params={}):
    wide_preprocessor = ChunkWidePreprocessor(
        wide_cols=cat_cols,
        n_chunks=n_chunks,
    )

    tab_preprocessor = ChunkTabPreprocessor(
        embed_cols=cat_cols,
        continuous_cols=num_cols,
        n_chunks=n_chunks,
        default_embed_dim=8,
        verbose=0,
        **tab_params,
    )

    text_preprocessor = ChunkTextPreprocessor(
        n_chunks=n_chunks, text_col=text_col, n_cpus=1, max_vocab=50, maxlen=10
    )

    img_preprocessor = ImagePreprocessor(
        img_col=img_col,
        img_path=img_folder,
    )

    for i, chunk in enumerate(
        pd.read_csv("/".join([data_folder, fname]), chunksize=chunksize)
    ):
        # the image processor does not need to be fitted before passed to the
        # loaders from folder
        wide_preprocessor.fit(chunk)
        tab_preprocessor.fit(chunk)
        text_preprocessor.fit(chunk)

    return wide_preprocessor, tab_preprocessor, text_preprocessor, img_preprocessor


def _build_data_mode_from_folder(
    wide_preprocessor,
    tab_preprocessor,
    text_preprocessor,
    img_preprocessor,
    target_col="target_regression",
):
    tab_from_folder = TabFromFolder(
        fname=fname,
        directory=data_folder,
        target_col=target_col,
        preprocessor=tab_preprocessor,
        img_col=img_col,
        text_col=text_col,
    )

    wide_from_folder = WideFromFolder(
        fname=fname,
        directory=data_folder,
        preprocessor=wide_preprocessor,
        reference=tab_from_folder,
    )

    text_from_folder = TextFromFolder(
        preprocessor=text_preprocessor,
    )

    img_from_folder = ImageFromFolder(preprocessor=img_preprocessor)

    return wide_from_folder, tab_from_folder, text_from_folder, img_from_folder


def _build_eval_and_test_data_mode_from_folder(
    wide_from_folder, tab_from_folder, eval_fname, test_fname
):
    eval_wide_from_folder = TabFromFolder(fname=eval_fname, reference=wide_from_folder)
    eval_tab_from_folder = TabFromFolder(fname=eval_fname, reference=tab_from_folder)

    test_wide_from_folder = TabFromFolder(
        fname=test_fname, reference=wide_from_folder, ignore_target=True
    )
    test_tab_from_folder = TabFromFolder(
        fname=test_fname, reference=tab_from_folder, ignore_target=True
    )

    return (
        eval_wide_from_folder,
        eval_tab_from_folder,
        test_wide_from_folder,
        test_tab_from_folder,
    )


def _buid_model(
    wide_preprocessor,
    tab_preprocessor,
    text_preprocessor,
    pred_dim=1,
    with_attention=False,
):
    wide = Wide(input_dim=wide_preprocessor.wide_dim, num_class=pred_dim)

    if with_attention:
        deeptabular = TabTransformer(
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            input_dim=8,
            n_heads=2,
            n_blocks=2,
        )
    else:
        deeptabular = TabMlp(
            mlp_hidden_dims=[16, 8],
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
        )

    basic_rnn = BasicRNN(
        vocab_size=len(text_preprocessor.vocab.itos),
        embed_dim=8,
        hidden_dim=8,
    )

    basic_cnn = Vision()

    model = WideDeep(
        wide=wide,
        deeptabular=deeptabular,
        deeptext=basic_rnn,
        deepimage=basic_cnn,
        num_class=pred_dim,
    )

    return model


@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
def test_trainer_from_loader_basic_inputs(objective):
    (
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        img_preprocessor,
    ) = _build_preprocessors()

    if objective == "regression":
        target_col = "target_regression"
    elif objective == "binary":
        target_col = "target_binary"
    else:
        target_col = "target_multiclass"
    (
        wide_from_folder,
        tab_from_folder,
        text_from_folder,
        img_from_folder,
    ) = _build_data_mode_from_folder(
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        img_preprocessor,
        target_col,
    )

    dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=wide_from_folder,
        tab_from_folder=tab_from_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    dataloader_from_folder = DataLoader(dataset_from_folder, batch_size=4)

    pred_dim = 1 if objective == "regression" or objective == "binary" else 3
    model = _buid_model(
        wide_preprocessor, tab_preprocessor, text_preprocessor, pred_dim=pred_dim
    )

    trainer = TrainerFromFolder(
        model,
        objective=objective,
        verbose=0,
    )

    trainer.fit(
        train_loader=dataloader_from_folder,
    )

    # simply assert that it has run and it has a history atttribute
    assert len(trainer.history) > 0 and "train_loss" in trainer.history.keys()


@pytest.mark.parametrize("pred_with_loader", [True, False])
def test_trainer_from_loader_with_valid_and_test(pred_with_loader):
    (
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        img_preprocessor,
    ) = _build_preprocessors()

    (
        wide_from_folder,
        tab_from_folder,
        text_from_folder,
        img_from_folder,
    ) = _build_data_mode_from_folder(
        wide_preprocessor, tab_preprocessor, text_preprocessor, img_preprocessor
    )

    (
        eval_wide_from_folder,
        eval_tab_from_folder,
        test_wide_from_folder,
        test_tab_from_folder,
    ) = _build_eval_and_test_data_mode_from_folder(
        wide_from_folder, tab_from_folder, fname, fname
    )

    train_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=wide_from_folder,
        tab_from_folder=tab_from_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    eval_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=eval_wide_from_folder,
        tab_from_folder=eval_tab_from_folder,
        reference=train_dataset_from_folder,
    )

    test_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=test_wide_from_folder,
        tab_from_folder=test_tab_from_folder,
        reference=train_dataset_from_folder,
    )

    train_dataloader_from_folder = DataLoader(train_dataset_from_folder, batch_size=4)
    eval_dataloader_from_folder = DataLoader(eval_dataset_from_folder, batch_size=4)
    test_dataloader_from_folder = DataLoader(test_dataset_from_folder, batch_size=4)

    model = _buid_model(wide_preprocessor, tab_preprocessor, text_preprocessor)

    trainer = TrainerFromFolder(
        model,
        objective="regression",
        verbose=0,
    )

    trainer.fit(
        train_loader=train_dataloader_from_folder,
        eval_loader=eval_dataloader_from_folder,
    )

    if pred_with_loader:
        preds = trainer.predict(test_loader=test_dataloader_from_folder)
    else:
        df = pd.read_csv("/".join([data_folder, fname]))
        X_test_wide = wide_preprocessor.transform(df)
        X_test_tab = tab_preprocessor.transform(df)
        X_test_text = text_preprocessor.transform(df)
        X_images = img_preprocessor.fit_transform(df)
        preds = trainer.predict(
            X_wide=X_test_wide,
            X_tab=X_test_tab,
            X_text=X_test_text,
            X_img=X_images,
            batch_size=4,
        )

    assert (
        preds.shape[0] == data_size
        and "train_loss" in trainer.history.keys()
        and "val_loss" in trainer.history.keys()
    )


@pytest.mark.parametrize(
    "tab_params",
    [
        {"with_attention": True, "with_cls_token": True},
        {"with_attention": True, "with_cls_token": False},
    ],
)
def test_trainer_from_loader_with_tab_params(tab_params):
    (
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        img_preprocessor,
    ) = _build_preprocessors(tab_params=tab_params)

    (
        wide_from_folder,
        tab_from_folder,
        text_from_folder,
        img_from_folder,
    ) = _build_data_mode_from_folder(
        wide_preprocessor, tab_preprocessor, text_preprocessor, img_preprocessor
    )

    (
        eval_wide_from_folder,
        eval_tab_from_folder,
        test_wide_from_folder,
        test_tab_from_folder,
    ) = _build_eval_and_test_data_mode_from_folder(
        wide_from_folder, tab_from_folder, fname, fname
    )

    train_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=wide_from_folder,
        tab_from_folder=tab_from_folder,
        text_from_folder=text_from_folder,
        img_from_folder=img_from_folder,
    )

    eval_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=eval_wide_from_folder,
        tab_from_folder=eval_tab_from_folder,
        reference=train_dataset_from_folder,
    )

    test_dataset_from_folder = WideDeepDatasetFromFolder(
        n_samples=data_size,
        wide_from_folder=test_wide_from_folder,
        tab_from_folder=test_tab_from_folder,
        reference=train_dataset_from_folder,
    )

    train_dataloader_from_folder = DataLoader(train_dataset_from_folder, batch_size=4)
    eval_dataloader_from_folder = DataLoader(eval_dataset_from_folder, batch_size=4)
    test_dataloader_from_folder = DataLoader(test_dataset_from_folder, batch_size=4)

    model = _buid_model(
        wide_preprocessor,
        tab_preprocessor,
        text_preprocessor,
        with_attention=tab_params["with_attention"],
    )

    trainer = TrainerFromFolder(
        model,
        objective="regression",
        verbose=1,
        finetune=True,
        callbacks=[EarlyStopping(patience=10)],  # any number higher than 2
    )

    trainer.fit(
        train_loader=train_dataloader_from_folder,
        eval_loader=eval_dataloader_from_folder,
        n_epochs=2,
    )

    preds = trainer.predict(test_loader=test_dataloader_from_folder)

    assert (
        preds.shape[0] == data_size
        and "train_loss" in trainer.history.keys()
        and "val_loss" in trainer.history.keys()
        and len(trainer.history["train_loss"]) == 2
    )
