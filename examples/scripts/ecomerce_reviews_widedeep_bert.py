from typing import List

import numpy as np
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
from pytorch_widedeep.models.tabular.mlp._layers import MLP


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
        attention_masks = []
        for text in texts:
            encoded_sent = self.tokenizer.encode_plus(
                text=self._pre_rules(text),
                add_special_tokens=True,
                max_length=self.max_length,
                padding=True,
                return_attention_mask=True,
            )

            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        return np.stack(input_ids), np.stack(attention_masks)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @staticmethod
    def _pre_rules(text):
        return fix_html(rm_useless_spaces(spec_add_spaces(text)))


class BertClassifier(nn.Module):
    def __init__(
        self,
        head_hidden_dim: List[int],
        model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = False,
        head_dropout: float = 0.0,
        num_class: int = 4,
    ):
        super(BertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)

        # need to edit this to add the output dim property
        classifier_dims = [768] + head_hidden_dim + [num_class]
        self.classifier = MLP(
            d_hidden=classifier_dims,
            activation="relu",
            dropout=0.1,
            batchnorm=False,
            batchnorm_last=False,
            linear_first=False,
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:

        # compute the mask here

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        return self.classifier(last_hidden_state_cls)


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

tokenizer = Tokenizer()
X_text_tr, X_masks_tr = tokenizer.fit_transform(train["review_text"].tolist())
X_text_te, X_masks_te = tokenizer.transform(test["review_text"].tolist())
