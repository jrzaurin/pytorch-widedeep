# HT models are available in the library, but in this script we show how
# to use a custom model with the WideDeep class.
import numpy as np
import torch
import pandas as pd
from torch import Tensor, nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.datasets import load_womens_ecommerce
from pytorch_widedeep.utils.fastai_transforms import (
    fix_html,
    spec_add_spaces,
    rm_useless_spaces,
)


class Tokenizer(object):
    def __init__(
        self,
        pretrained_tokenizer="distilbert-base-uncased",
        do_lower_case=True,
        max_length=90,
    ):
        super(Tokenizer, self).__init__()
        self.pretrained_tokenizer = pretrained_tokenizer
        self.do_lower_case = do_lower_case
        self.max_length = max_length

    def fit(self, texts):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.pretrained_tokenizer, do_lower_case=self.do_lower_case
        )

        return self

    def transform(self, texts):
        input_ids = []
        for text in texts:
            encoded_sent = self.tokenizer.encode_plus(
                text=self._pre_rules(text),
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

            input_ids.append(encoded_sent.get("input_ids"))
        return np.stack(input_ids)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @staticmethod
    def _pre_rules(text):
        return fix_html(rm_useless_spaces(spec_add_spaces(text)))


class CustomBertModel(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = False,
    ):
        super(CustomBertModel, self).__init__()

        self.bert = DistilBertModel.from_pretrained(
            model_name,
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, X_inp: Tensor) -> Tensor:
        attn_mask = (X_inp != 0).type(torch.int8)
        outputs = self.bert(input_ids=X_inp, attention_mask=attn_mask)
        return outputs[0][:, 0, :]

    @property
    def output_dim(self) -> int:
        return 768


df: pd.DataFrame = load_womens_ecommerce(as_frame=True)

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

tokenizer = Tokenizer()
X_text_tr = tokenizer.fit_transform(train["review_text"].tolist())
X_text_te = tokenizer.transform(test["review_text"].tolist())

bert_model = CustomBertModel(freeze_bert=True)
model = WideDeep(
    deeptext=bert_model,
    head_hidden_dims=[256, 128, 64],
    pred_dim=4,
)

trainer = Trainer(
    model,
    objective="multiclass",
    metrics=[Accuracy, F1Score(average=True)],
)

trainer.fit(
    X_text=X_text_tr,
    target=train.rating.values,
    n_epochs=5,
    batch_size=64,
)

preds_text = trainer.predict_proba(X_text=X_text_te)
pred_text_class = np.argmax(preds_text, 1)

acc_text = accuracy_score(test.rating, pred_text_class)
f1_text = f1_score(test.rating, pred_text_class, average="weighted")
prec_text = precision_score(test.rating, pred_text_class, average="weighted")
rec_text = recall_score(test.rating, pred_text_class, average="weighted")
cm_text = confusion_matrix(test.rating, pred_text_class)
