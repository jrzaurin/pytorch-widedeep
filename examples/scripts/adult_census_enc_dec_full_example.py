import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import EncoderDecoderTrainer

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    df = load_adult(as_frame=True)
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
        cat_embed_cols=cat_embed_cols, continuous_cols=continuous_cols, scale=True
    )
    X_tab = tab_preprocessor.fit_transform(df_tr)
    target = df_tr[target_col].values

    # We define a model that will act as the encoder in the encoder/decoder
    # architecture. This could be any of: TabMlp, TabResnet or TabNet
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
    )

    # If we do not pass a custom decoder, which is perfectly possible via the
    # decoder param (see the docs or the examples notebooks, the
    # EncoderDecoderTrainer will automatically build a decoder which will be
    # the 'mirror' image of the encoder
    encoder_decoder_trainer = EncoderDecoderTrainer(encoder=tab_mlp)
    encoder_decoder_trainer.pretrain(X_tab, n_epochs=5, batch_size=256)

    # At this point we have two options:

    # 1. We can save the pretrained model for later use
    encoder_decoder_trainer.save(
        path="pretrained_weights", model_filename="encoder_decoder_model.pt"
    )

    # some time has passed, we load the model with torch as usual:
    encoder_decoder_model = torch.load("pretrained_weights/encoder_decoder_model.pt")

    # NOW, AND THIS IS IMPORTANT! We have loaded the encoder AND the decoder.
    # To proceed to the supervised training we ONLY need the encoder
    pretrained_encoder = encoder_decoder_model.encoder

    # and as always, ANY supervised model in this library has to go throuth the WideDeep class:
    model = WideDeep(deeptabular=pretrained_encoder)
    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy])

    trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256)

    # And, you know...we get a test metric
    X_tab_te = tab_preprocessor.transform(df_te)
    target_te = df_te[target_col].values

    preds = trainer.predict(X_tab=X_tab_te)
    test_acc = accuracy_score(target_te, preds)

    # # 2. We could just move on without saving (if you are that wild), in which case, simply, after calling

    # encoder_decoder_trainer.pretrain(X_tab, n_epochs=5, batch_size=256)

    # # you just have to

    # model = WideDeep(deeptabular=tab_mlp)
    # trainer = Trainer(model=model, objective="binary", metrics=[Accuracy])

    # trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256)

    # # And, you know...we get a test metric
    # X_tab_te = tab_preprocessor.transform(df_te)
    # target_te = df_te[target_col].values

    # preds = trainer.predict(X_tab=X_tab_te)
    # test_acc = accuracy_score(target_te, preds)
