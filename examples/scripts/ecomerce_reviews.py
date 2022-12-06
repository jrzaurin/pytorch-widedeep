import numpy as np
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from pytorch_widedeep.utils import Tokenizer, LabelEncoder
from pytorch_widedeep.datasets import load_womens_ecommerce

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

train, valid = train_test_split(df, train_size=0.8, random_state=1, stratify=df.rating)
valid, test = train_test_split(
    valid, train_size=0.5, random_state=1, stratify=valid.rating
)


###############################################################################
# The simplest thing: tokenization + tf-idf
tok = Tokenizer()
tok_reviews_tr = tok.process_all(train.review_text.tolist())
tok_reviews_va = tok.process_all(valid.review_text.tolist())
vectorizer = TfidfVectorizer(
    max_features=5000, preprocessor=lambda x: x, tokenizer=lambda x: x, min_df=5
)

X_text_tr = vectorizer.fit_transform(tok_reviews_tr)
X_text_va = vectorizer.transform(tok_reviews_va)

lgbtrain_text = lgbDataset(
    X_text_tr,
    train.rating.values,
    free_raw_data=False,
)

lgbvalid_text = lgbDataset(
    X_text_va,
    valid.rating.values,
    reference=lgbtrain_text,
    free_raw_data=False,
)

model_text = lgb.train(
    {"objective": "multiclass", "num_classes": 4},
    lgbtrain_text,
    valid_sets=[lgbvalid_text, lgbtrain_text],
    valid_names=["test", "train"],
    early_stopping_rounds=50,
)

preds_text = model_text.predict(X_text_va)
pred_text_class = np.argmax(preds_text, 1)

acc_text = accuracy_score(lgbvalid_text.label, pred_text_class)
f1_text = f1_score(lgbvalid_text.label, pred_text_class, average="weighted")
prec_text = precision_score(lgbvalid_text.label, pred_text_class, average="weighted")
rec_text = recall_score(lgbvalid_text.label, pred_text_class, average="weighted")
cm_text = confusion_matrix(lgbvalid_text.label, pred_text_class)


###############################################################################
# Text + Tabular

tab_cols = [
    "age",
    "division_name",
    "department_name",
    "class_name",
]

for tab_df in [train, valid]:
    for c in ["division_name", "department_name", "class_name"]:
        tab_df[c] = tab_df[c].str.lower()

le = LabelEncoder(columns_to_encode=["division_name", "department_name", "class_name"])
train_tab_le = le.fit_transform(train)
valid_tab_le = le.transform(valid)

X_tab_tr = csr_matrix(train_tab_le[tab_cols].values)
X_tab_va = csr_matrix(valid_tab_le[tab_cols].values)

X_tab_text_tr = hstack((X_tab_tr, X_text_tr))
X_tab_text_va = hstack((X_tab_va, X_text_va))

lgbtrain_tab_text = lgbDataset(
    X_tab_text_tr,
    train.rating.values,
    categorical_feature=[0, 1, 2, 3],
    free_raw_data=False,
)

lgbvalid_tab_text = lgbDataset(
    X_tab_text_va,
    valid.rating.values,
    reference=lgbtrain_tab_text,
    free_raw_data=False,
)

model_tab_text = lgb.train(
    {"objective": "multiclass", "num_classes": 4},
    lgbtrain_tab_text,
    valid_sets=[lgbtrain_tab_text, lgbvalid_tab_text],
    valid_names=["test", "train"],
    early_stopping_rounds=50,
)

preds_tab_text = model_tab_text.predict(X_tab_text_va)
preds_tab_text_class = np.argmax(preds_tab_text, 1)

acc_tab_text = accuracy_score(lgbvalid_tab_text.label, preds_tab_text_class)
f1_tab_text = f1_score(
    lgbvalid_tab_text.label, preds_tab_text_class, average="weighted"
)
prec_tab_text = precision_score(
    lgbvalid_tab_text.label, preds_tab_text_class, average="weighted"
)
rec_tab_text = recall_score(
    lgbvalid_tab_text.label, preds_tab_text_class, average="weighted"
)
cm_tab_text = confusion_matrix(lgbvalid_tab_text.label, preds_tab_text_class)


# ###############################################################################
# #  randon and most popular pred (need to do this last one)
# random_preds = np.random.choice(
#     np.arange(4),
#     size=len(valid),
#     p=(valid.rating.value_counts() / len(valid)).values[::-1],
# )

# preds = model.predict(X_va)
# pred_class = np.argmax(preds, 1)

# rand_acc = accuracy_score(lgbvalid.label, random_preds)
# rand_f1 = f1_score(lgbvalid.label, random_preds, average="weighted")
# rand_prec = precision_score(lgbvalid.label, random_preds, average="weighted")
# rand_rec = recall_score(lgbvalid.label, random_preds, average="weighted")
# rand_cm = confusion_matrix(lgbvalid.label, random_preds)

# acc = accuracy_score(lgbvalid.label, pred_class)
# f1 = f1_score(lgbvalid.label, pred_class, average="weighted")
# prec = precision_score(lgbvalid.label, pred_class, average="weighted")
# rec = recall_score(lgbvalid.label, pred_class, average="weighted")
# cm = confusion_matrix(lgbvalid.label, pred_class)
