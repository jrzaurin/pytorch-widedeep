import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import BasicRNN, WideDeep
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.datasets import load_womens_ecommerce
from pytorch_widedeep.callbacks import LRHistory
from pytorch_widedeep.initializers import XavierNormal, KaimingNormal
from pytorch_widedeep.preprocessing import TextPreprocessor

df: pd.DataFrame = load_womens_ecommerce(as_frame=True)

df.columns = [c.replace(" ", "_").lower() for c in df.columns]

# classes from [0,num_class)
df["rating"] = (df["rating"] - 1).astype("int64")

# group reviews with 1 and 2 scores into one class
df.loc[df.rating == 0, "rating"] = 1

# and back again to [0,num_class)
df["rating"] = (df["rating"] - 1).astype("int64")

# drop rows with no title
df = df[~df.title.isna()].reset_index(drop=True)

# drop short reviews
df = df[~df.review_text.isna()]
df["review_length"] = df.review_text.apply(lambda x: len(x.split(" ")))
df = df[df.review_length >= 5]
df = df.drop("review_length", axis=1).reset_index(drop=True)

# we are not going to carry any hyperparameter optimization so no need of
# validation set
train, test = train_test_split(df, train_size=0.8, random_state=1, stratify=df.rating)

text_preprocessor_review = TextPreprocessor(
    text_col="review_text", max_vocab=5000, min_freq=5, maxlen=90, n_cpus=1
)

text_preprocessor_title = TextPreprocessor(
    text_col="title", max_vocab=1000, min_freq=5, maxlen=10, n_cpus=1
)

X_text_review_tr = text_preprocessor_review.fit_transform(train)
X_text_review_te = text_preprocessor_review.transform(test)

X_text_title_tr = text_preprocessor_title.fit_transform(train)
X_text_title_te = text_preprocessor_title.transform(test)

basic_rnn_review = BasicRNN(
    vocab_size=len(text_preprocessor_review.vocab.itos),
    embed_dim=300,
    hidden_dim=64,
    n_layers=3,
    rnn_dropout=0.2,
    head_hidden_dims=[32],
)

basic_rnn_title = BasicRNN(
    vocab_size=len(text_preprocessor_title.vocab.itos),
    embed_dim=50,
    hidden_dim=32,
    n_layers=1,
    head_hidden_dims=[8],
)

model = WideDeep(deeptext=[basic_rnn_review, basic_rnn_title], pred_dim=4)

review_opt = torch.optim.Adam(model.deeptext[0].parameters(), lr=0.01)
title_opt = torch.optim.Adam(model.deeptext[1].parameters(), lr=0.05)

review_sch = torch.optim.lr_scheduler.StepLR(review_opt, step_size=2)
title_sch = torch.optim.lr_scheduler.StepLR(title_opt, step_size=3)

optimizers = {"deeptext": [review_opt, title_opt]}
schedulers = {"deeptext": [review_sch, title_sch]}
initializers = {"deeptext": [XavierNormal, KaimingNormal]}

trainer = Trainer(
    model,
    objective="multiclass",
    optimizers=optimizers,
    lr_schedulers=schedulers,
    initializers=initializers,
    metrics=[Accuracy, F1Score(average=True)],
    callbacks=[LRHistory(n_epochs=10)],
)

trainer.fit(
    X_text=[X_text_review_tr, X_text_title_tr],
    target=train.rating.values,
    n_epochs=10,
    batch_size=256,
)

preds_text = trainer.predict_proba(X_text=[X_text_review_te, X_text_title_te])
pred_text_class = np.argmax(preds_text, 1)

acc_text = accuracy_score(test.rating, pred_text_class)
f1_text = f1_score(test.rating, pred_text_class, average="weighted")
prec_text = precision_score(test.rating, pred_text_class, average="weighted")
rec_text = recall_score(test.rating, pred_text_class, average="weighted")
cm_text = confusion_matrix(test.rating, pred_text_class)
