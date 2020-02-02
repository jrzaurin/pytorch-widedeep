import pandas as pd
import torch

from torchvision.transforms import ToTensor, Normalize

from pytorch_widedeep.preprocessing import (
    WidePreprocessor,
    DeepPreprocessor,
    TextPreprocessor,
    ImagePreprocessor,
)
from pytorch_widedeep.models import Wide, DeepDense, DeepText, DeepImage, WideDeep
from pytorch_widedeep.initializers import KaimingNormal
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.optim import RAdam


use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = pd.read_csv("data/airbnb/airbnb_sample.csv")

    crossed_cols = (["property_type", "room_type"],)
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
    word_vectors_path = "data/glove.6B/glove.6B.100d.txt"
    img_col = "id"
    img_path = "data/airbnb/property_picture"
    target = "yield"

    target = df[target].values

    prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)

    prepare_deep = DeepPreprocessor(
        embed_cols=cat_embed_cols, continuous_cols=continuous_cols
    )
    X_deep = prepare_deep.fit_transform(df)

    text_processor = TextPreprocessor(
        word_vectors_path=word_vectors_path, text_col=text_col
    )
    X_text = text_processor.fit_transform(df)

    image_processor = ImagePreprocessor(img_col=img_col, img_path=img_path)
    X_images = image_processor.fit_transform(df)

    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
    deepdense = DeepDense(
        hidden_layers=[64, 32],
        dropout=[0.2, 0.2],
        deep_column_idx=prepare_deep.deep_column_idx,
        embed_input=prepare_deep.embeddings_input,
        continuous_cols=continuous_cols,
    )
    deeptext = DeepText(
        vocab_size=len(text_processor.vocab.itos),
        hidden_dim=64,
        n_layers=3,
        rnn_dropout=0.5,
        padding_idx=1,
        embedding_matrix=text_processor.embedding_matrix,
    )
    deepimage = DeepImage(pretrained=True, head_layers=None)
    model = WideDeep(
        wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage
    )

    wide_opt = torch.optim.Adam(model.wide.parameters())
    deep_opt = torch.optim.Adam(model.deepdense.parameters())
    text_opt = RAdam(model.deeptext.parameters())
    img_opt = RAdam(model.deepimage.parameters())

    wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
    text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
    img_sch = torch.optim.lr_scheduler.StepLR(img_opt, step_size=3)

    optimizers = {
        "wide": wide_opt,
        "deepdense": deep_opt,
        "deeptext": text_opt,
        "deepimage": img_opt,
    }
    schedulers = {
        "wide": wide_sch,
        "deepdense": deep_sch,
        "deeptext": text_sch,
        "deepimage": img_sch,
    }
    initializers = {
        "wide": KaimingNormal,
        "deepdense": KaimingNormal,
        "deeptext": KaimingNormal,
        "deepimage": KaimingNormal,
    }
    mean = [0.406, 0.456, 0.485]  # BGR
    std = [0.225, 0.224, 0.229]  # BGR
    transforms = [ToTensor, Normalize(mean=mean, std=std)]
    callbacks = [EarlyStopping, ModelCheckpoint(filepath="model_weights/wd_out.pt")]

    model.compile(
        method="regression",
        initializers=initializers,
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=callbacks,
        transforms=transforms,
    )

    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        X_text=X_text,
        X_img=X_images,
        target=target,
        n_epochs=1,
        batch_size=32,
        val_split=0.2,
    )

    # # With warm_up
    # child = list(model.deepimage.children())[0]
    # img_layers = list(child.backbone.children())[4:8] + [list(model.deepimage.children())[1]]
    # img_layers = img_layers[::-1]

    # model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_images,
    #     target=target, n_epochs=1, batch_size=32, val_split=0.2, warm_up=True,
    #     warm_epochs=1, warm_deepimage_gradual=True, warm_deepimage_layers=img_layers,
    #     warm_deepimage_max_lr=0.01, warm_routine='howard')
