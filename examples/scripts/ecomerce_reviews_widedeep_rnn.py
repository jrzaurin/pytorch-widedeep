import numpy as np
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
from pytorch_widedeep.preprocessing import TextPreprocessor

df = load_womens_ecommerce(as_frame=True)

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

# we are not going to carry any hyperparameter optimization so no need of
# validation set
train, test = train_test_split(df, train_size=0.8, random_state=1, stratify=df.rating)

text_preprocessor = TextPreprocessor(
    text_col="review_text", max_vocab=5000, min_freq=5, maxlen=90, n_cpus=1
)

X_text_tr = text_preprocessor.fit_transform(train)
X_text_te = text_preprocessor.transform(test)

basic_rnn = BasicRNN(
    vocab_size=len(text_preprocessor.vocab.itos),
    embed_dim=300,
    hidden_dim=64,
    n_layers=3,
    rnn_dropout=0.2,
    head_hidden_dims=[32],
)


model = WideDeep(deeptext=basic_rnn, pred_dim=4)

trainer = Trainer(
    model,
    objective="multiclass",
    metrics=[Accuracy, F1Score(average=True)],
)

trainer.fit(
    X_text=X_text_tr,
    target=train.rating.values,
    n_epochs=5,
    batch_size=256,
)


preds_text = trainer.predict_proba(X_text=X_text_te)
pred_text_class = np.argmax(preds_text, 1)

acc_text = accuracy_score(test.rating, pred_text_class)
f1_text = f1_score(test.rating, pred_text_class, average="weighted")
prec_text = precision_score(test.rating, pred_text_class, average="weighted")
rec_text = recall_score(test.rating, pred_text_class, average="weighted")
cm_text = confusion_matrix(test.rating, pred_text_class)
