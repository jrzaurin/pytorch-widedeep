import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import HFModel, WideDeep
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.datasets import load_womens_ecommerce
from pytorch_widedeep.preprocessing import HFPreprocessor as HFTokenizer

df: pd.DataFrame = load_womens_ecommerce(as_frame=True)  # type: ignore

df.columns = [c.replace(" ", "_").lower() for c in df.columns]

# classes from [0,num_class)
df["rating"] = (df["rating"] - 1).astype("int64")

# group reviews with 1 and 2 scores into one class
df.loc[df.rating == 0, "rating"] = 1

# and back again to [0,num_class)
df["rating"] = (df["rating"] - 1).astype("int64")

# drop short reviews
df = df[~df.review_text.isna()]
df["review_length"] = df.review_text.apply(lambda x: len(x.split(" ")))
df = df[df.review_length >= 5]
df = df.drop("review_length", axis=1).reset_index(drop=True)

if not torch.cuda.is_available():
    # stratified sample to the minimum and then sample at random
    # rating
    # 3    12515
    # 2     4904
    # 1     2820
    # 0     2369
    df = (
        df.groupby("rating", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 2369)))
        .sample(1000)
    )
    model_names = [
        "distilbert-base-uncased",
    ]
else:
    model_names = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "FacebookAI/roberta-base",
        "albert-base-v2",
        "google/electra-base-discriminator",
    ]

train, test = train_test_split(df, train_size=0.8, random_state=1, stratify=df.rating)

for model_name in model_names:
    print(f"Training model: {model_name}")
    tokenizer = HFTokenizer(
        model_name=model_name,
        text_col="review_text",
        num_workers=1,
        encode_params={
            "max_length": 90,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
        },
    )

    X_text_tr = tokenizer.fit_transform(train)
    X_text_te = tokenizer.transform(test)

    hf_model = HFModel(
        model_name=model_name,
        use_cls_token=True,
    )

    model = WideDeep(
        deeptext=hf_model,
        head_hidden_dims=[256, 64],
        pred_dim=4,
    )

    trainer = Trainer(
        model,
        objective="multiclass",
        metrics=[Accuracy(), F1Score(average=True)],
    )

    trainer.fit(
        X_text=X_text_tr,
        target=train.rating.values,
        n_epochs=1,
        batch_size=64,
    )

    preds_text = trainer.predict_proba(X_text=X_text_te)
    pred_text_class = np.argmax(preds_text, 1)

    acc_text = accuracy_score(test.rating, pred_text_class)
    f1_text = f1_score(test.rating, pred_text_class, average="weighted")
    print(f"Accuracy: {acc_text:.4f}")
    print(f"F1: {f1_text:.4f}")
