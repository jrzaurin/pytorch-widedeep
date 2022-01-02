from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torchvision.transforms import ToTensor, Normalize

import pytorch_widedeep as wd
from pytorch_widedeep.models import TabResnet  # noqa: F401
from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    Vision,
    BasicRNN,
    WideDeep,
    AttentiveRNN,
    StackedAttentiveRNN,
)
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.initializers import KaimingNormal
from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    WidePreprocessor,
    ImagePreprocessor,
)

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    DATA_PATH = Path("../tmp_data")

    df = pd.read_csv(DATA_PATH / "airbnb/airbnb_sample.csv")

    crossed_cols = [("property_type", "room_type")]
    already_dummies = [c for c in df.columns if "amenity" in c] + ["has_house_rules"]
    wide_cols = [
        "is_location_exact",
        "property_type",
        "room_type",
        "host_gender",
        "instant_bookable",
    ] + already_dummies
    cat_embed_cols = [(c, 16) for c in df.columns if "catg" in c] + [
        ("neighbourhood_cleansed", 64),
        ("cancellation_policy", 16),
    ]
    continuous_cols = ["latitude", "longitude", "security_deposit", "extra_people"]
    already_standard = ["latitude", "longitude"]
    text_col = "description"
    word_vectors_path = str(DATA_PATH / "glove.6B/glove.6B.100d.txt")
    img_col = "id"
    img_path = str(DATA_PATH / "airbnb/property_picture")
    target = "yield"

    target = df[target].values

    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = wide_preprocessor.fit_transform(df)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,  # type: ignore[arg-type]
        continuous_cols=continuous_cols,
        already_standard=already_standard,
    )
    X_tab = tab_preprocessor.fit_transform(df)

    image_processor = ImagePreprocessor(img_col=img_col, img_path=img_path)
    X_images = image_processor.fit_transform(df)
    deepimage = Vision(
        pretrained_model_name="resnet18", n_trainable=0, head_hidden_dims=[200, 100]
    )

    text_processor = TextPreprocessor(
        word_vectors_path=word_vectors_path, text_col=text_col
    )
    X_text = text_processor.fit_transform(df)
    deeptext = BasicRNN(
        vocab_size=len(text_processor.vocab.itos),
        hidden_dim=64,
        n_layers=3,
        bidirectional=True,
        rnn_dropout=0.5,
        padding_idx=1,
        embed_matrix=text_processor.embedding_matrix,
        head_hidden_dims=[100, 50],
    )
    # deeptext = AttentiveRNN(
    #     vocab_size=len(text_processor.vocab.itos),
    #     hidden_dim=64,
    #     n_layers=3,
    #     bidirectional=True,
    #     rnn_dropout=0.5,
    #     padding_idx=1,
    #     embed_matrix=text_processor.embedding_matrix,
    #     with_attention=True,
    #     head_hidden_dims=[100, 50],
    # )
    # deeptext = StackedAttentiveRNN(
    #     vocab_size=len(text_processor.vocab.itos),
    #     embed_matrix=text_processor.embedding_matrix,
    #     hidden_dim=64,
    #     bidirectional=True,
    #     padding_idx=1,
    #     with_addnorm=True,
    #     head_hidden_dims=[100, 50],
    # )

    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
    deepdense = TabMlp(
        mlp_hidden_dims=[64, 32],
        mlp_dropout=[0.2, 0.2],
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=continuous_cols,
    )

    model = WideDeep(
        wide=wide,
        deeptabular=deepdense,
        deeptext=deeptext,
        deepimage=deepimage,
    )

    wide_opt = torch.optim.Adam(model.wide.parameters(), lr=0.01)
    deep_opt = torch.optim.Adam(model.deeptabular.parameters())
    text_opt = torch.optim.AdamW(model.deeptext.parameters())
    img_opt = torch.optim.AdamW(model.deepimage.parameters())

    wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
    text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
    img_sch = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)

    optimizers = {
        "wide": wide_opt,
        "deeptabular": deep_opt,
        "deeptext": text_opt,
        "deepimage": img_opt,
    }
    schedulers = {
        "wide": wide_sch,
        "deeptabular": deep_sch,
        "deeptext": text_sch,
        "deepimage": img_sch,
    }
    initializers = {
        "wide": KaimingNormal,
        "deeptabular": KaimingNormal,
        "deeptext": KaimingNormal,
        "deepimage": KaimingNormal,
    }
    mean = [0.406, 0.456, 0.485]  # BGR
    std = [0.225, 0.224, 0.229]  # BGR
    transforms = [ToTensor, Normalize(mean=mean, std=std)]
    callbacks = [EarlyStopping, ModelCheckpoint(filepath="model_weights/wd_out.pt")]

    trainer = wd.Trainer(
        model,
        objective="regression",
        initializers=initializers,
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=callbacks,
        transforms=transforms,
    )

    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_images,
        target=target,
        n_epochs=1,
        batch_size=32,
        val_split=0.2,
    )

    # # With warm_up
    # child = list(trainer.model.deepimage.children())[0]
    # img_layers = list(child.backbone.children())[4:8] + [
    #     list(trainer.model.deepimage.children())[1]
    # ]
    # img_layers = img_layers[::-1]

    # trainer.fit(
    #     X_wide=X_wide,
    #     X_tab=X_tab,
    #     X_text=X_text,
    #     X_img=X_images,
    #     target=target,
    #     n_epochs=1,
    #     batch_size=32,
    #     val_split=0.2,
    #     warm_up=True,
    #     warm_epochs=1,
    #     warm_deepimage_gradual=True,
    #     warm_deepimage_layers=img_layers,
    #     warm_deepimage_max_lr=0.01,
    #     warm_routine="howard",
    # )
