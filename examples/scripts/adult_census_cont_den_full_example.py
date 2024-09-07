import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, TabTransformer
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import ContrastiveDenoisingTrainer

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    df: pd.DataFrame = load_adult(as_frame=True)
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop("income", axis=1, inplace=True)

    # one could chose to use a validation set for early stopping, hyperparam
    # optimization, etc. This is just an example, so we simply use train/test
    # split
    df_tr, df_te = train_test_split(df, test_size=0.2, stratify=df.income_label)

    cat_embed_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital_gain",
        "capital_loss",
        "native_country",
    ]
    continuous_cols = ["age", "hours_per_week"]
    target_col = "income_label"

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        scale=True,
        with_attention=True,
        with_cls_token=True,  # this is optional
    )
    X_tab = tab_preprocessor.fit_transform(df_tr)
    target = df_tr[target_col].values

    # here the model can be any attention-based model: SAINT,
    # FTTransformer, TabFastFormer, TabTransformer, SelfAttentionMLP and
    # ContextAttentionMLP,
    tab_transformer = TabTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        embed_continuous_method="standard",
        n_blocks=3,
    )

    # ContrastiveDenoisingTrainer implementes the procedure described in the
    # SAINT paper. See the notebooks for a more in -depth explanation
    contrastive_denoising_trainer = ContrastiveDenoisingTrainer(
        model=tab_transformer,
        preprocessor=tab_preprocessor,
        projection_head1_dims=[
            32,
            16,
        ],  # These params (projection_head_dims 1 and 2, and others) involve that you know your architecture
        projection_head2_dims=[32, 16],
    )
    contrastive_denoising_trainer.pretrain(X_tab, n_epochs=3, batch_size=256)

    # At this point we have two options:

    # 1. We can save the pretrained model for later use
    contrastive_denoising_trainer.save(
        path="pretrained_weights", model_filename="contrastive_denoising_model.pt"
    )

    # some time has passed, we load the model with torch as usual:
    contrastive_denoising_model = torch.load(
        "pretrained_weights/contrastive_denoising_model.pt"
    )

    # NOW, AND THIS IS IMPORTANT! We have loaded the entire contrastive,
    # denoising. To proceed to the supervised training we ONLY need the
    # attention-based model, which is the 'model' attribute
    pretrained_model = contrastive_denoising_model.model

    # and as always, ANY supervised model in this library has to go throuth the WideDeep class:
    model = WideDeep(deeptabular=pretrained_model)
    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy])

    trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256)

    # And, you know...we get a test metric
    X_tab_te = tab_preprocessor.transform(df_te)
    target_te = df_te[target_col].values

    preds = trainer.predict(X_tab=X_tab_te)
    test_acc = accuracy_score(target_te, preds)

    # # 2. We could just move on without saving (if you are that wild), in which case, simply, after calling

    # contrastive_denoising_trainer.pretrain(X_tab, n_epochs=3, batch_size=256)

    # # you just have to

    # model = WideDeep(deeptabular=tab_transformer)
    # trainer = Trainer(model=model, objective="binary", metrics=[Accuracy])

    # trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256)

    # # And, you know...we get a test metric
    # X_tab_te = tab_preprocessor.transform(df_te)
    # target_te = df_te[target_col].values

    # preds = trainer.predict(X_tab=X_tab_te)
    # test_acc = accuracy_score(target_te, preds)
