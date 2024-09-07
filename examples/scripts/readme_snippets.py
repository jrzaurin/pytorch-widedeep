# this is a script to illustrate the different architecture combinations that
# can be built with pytorch-widedeep in their simplest form. The script is
# not intended to be executed, but to be used as a reference

import os
import random

import numpy as np
import torch
import pandas as pd
from PIL import Image
from faker import Faker

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, TabMlp, Vision, BasicRNN, WideDeep, ModelFuser
from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    WidePreprocessor,
    ImagePreprocessor,
)
from pytorch_widedeep.losses_multitarget import MultiTargetClassificationLoss
from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent


def create_and_save_random_image(image_number, size=(32, 32)):

    if not os.path.exists("images"):
        os.makedirs("images")

    array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

    image = Image.fromarray(array)

    image_name = f"image_{image_number}.png"
    image.save(os.path.join("images", image_name))

    return image_name


fake = Faker()

cities = ["New York", "Los Angeles", "Chicago", "Houston"]
names = ["Alice", "Bob", "Charlie", "David", "Eva"]

data = {
    "city": [random.choice(cities) for _ in range(100)],
    "name": [random.choice(names) for _ in range(100)],
    "age": [random.uniform(18, 70) for _ in range(100)],
    "height": [random.uniform(150, 200) for _ in range(100)],
    "sentence": [fake.sentence() for _ in range(100)],
    "other_sentence": [fake.sentence() for _ in range(100)],
    "image_name": [create_and_save_random_image(i) for i in range(100)],
    "target": [random.choice([0, 1]) for _ in range(100)],
}

df = pd.DataFrame(data)

# 1. Wide and Tabular data

# Wide
wide_cols = ["city"]
crossed_cols = [("city", "name")]
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = wide_preprocessor.fit_transform(df)
wide = Wide(input_dim=np.unique(X_wide).shape[0])

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

model = WideDeep(wide=wide, deeptabular=tab_mlp)

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_wide=X_wide,
    X_tab=X_tab,
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)

# 2. Tabular and Text data

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

# Text
text_preprocessor = TextPreprocessor(
    text_col="sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text = text_preprocessor.fit_transform(df)
rnn = BasicRNN(
    vocab_size=len(text_preprocessor.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

model = WideDeep(deeptabular=tab_mlp, deeptext=rnn)

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_tab=X_tab,
    X_text=X_text,
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)

# 3. Tabular and text with a FC head on top via the 'head_hidden_dims' param in WideDeep

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

# Text
text_preprocessor = TextPreprocessor(
    text_col="sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text = text_preprocessor.fit_transform(df)
rnn = BasicRNN(
    vocab_size=len(text_preprocessor.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

model = WideDeep(deeptabular=tab_mlp, deeptext=rnn, head_hidden_dims=[32, 16])

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_tab=X_tab,
    X_text=X_text,
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)

# 4. Tabular with multiple text columns that are passed directly to WideDeep

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

# Text
text_preprocessor_1 = TextPreprocessor(
    text_col="sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_1 = text_preprocessor_1.fit_transform(df)
text_preprocessor_2 = TextPreprocessor(
    text_col="other_sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_2 = text_preprocessor_2.fit_transform(df)
rnn_1 = BasicRNN(
    vocab_size=len(text_preprocessor_1.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)
rnn_2 = BasicRNN(
    vocab_size=len(text_preprocessor_2.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

model = WideDeep(
    deeptabular=tab_mlp,
    deeptext=[rnn_1, rnn_2],
)

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_tab=X_tab,
    X_text=[X_text_1, X_text_2],
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)

# 5. Tabular data with multiple text columns that are fused via a ModelFuser

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

# Text
text_preprocessor_1 = TextPreprocessor(
    text_col="sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_1 = text_preprocessor_1.fit_transform(df)
text_preprocessor_2 = TextPreprocessor(
    text_col="other_sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_2 = text_preprocessor_2.fit_transform(df)

rnn_1 = BasicRNN(
    vocab_size=len(text_preprocessor_1.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)
rnn_2 = BasicRNN(
    vocab_size=len(text_preprocessor_2.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

models_fuser = ModelFuser(
    models=[rnn_1, rnn_2],
    fusion_method="mult",
)

model = WideDeep(
    deeptabular=tab_mlp,
    deeptext=models_fuser,
)

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_tab=X_tab,
    X_text=[X_text_1, X_text_2],
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)

# 6. Tabular with Multiple text columns, with an image column. The text
# columns fused via a ModelFuser and then all fused via the deephead
# paramenter in WideDeep

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[16, 8],
)

# Text
text_preprocessor_1 = TextPreprocessor(
    text_col="sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_1 = text_preprocessor_1.fit_transform(df)
text_preprocessor_2 = TextPreprocessor(
    text_col="other_sentence", maxlen=20, max_vocab=100, n_cpus=1
)
X_text_2 = text_preprocessor_2.fit_transform(df)

rnn_1 = BasicRNN(
    vocab_size=len(text_preprocessor_1.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

rnn_2 = BasicRNN(
    vocab_size=len(text_preprocessor_2.vocab.itos),
    embed_dim=16,
    hidden_dim=8,
    n_layers=1,
)

models_fuser = ModelFuser(
    models=[rnn_1, rnn_2],
    fusion_method="mult",
)

# Image

image_preprocessor = ImagePreprocessor(img_col="image_name", img_path="images")
X_img = image_preprocessor.fit_transform(df)

vision = Vision(pretrained_model_setup="resnet18", head_hidden_dims=[16, 8])


# deephead
class MyModelFuser(BaseWDModelComponent):
    def __init__(
        self,
        tab_incoming_dim: int,
        text_incoming_dim: int,
        image_incoming_dim: int,
        output_units: int,
    ):

        super(MyModelFuser, self).__init__()

        self.tab_incoming_dim = tab_incoming_dim
        self.text_incoming_dim = text_incoming_dim
        self.image_incoming_dim = image_incoming_dim
        self.output_units = output_units
        self.text_and_image_fuser = torch.nn.Sequential(
            torch.nn.Linear(text_incoming_dim + image_incoming_dim, output_units),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(output_units + tab_incoming_dim, output_units * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(output_units * 4, output_units),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        tab_slice = slice(0, self.tab_incoming_dim)
        text_slice = slice(
            self.tab_incoming_dim, self.tab_incoming_dim + self.text_incoming_dim
        )
        image_slice = slice(
            self.tab_incoming_dim + self.text_incoming_dim,
            self.tab_incoming_dim + self.text_incoming_dim + self.image_incoming_dim,
        )
        X_tab = X[:, tab_slice]
        X_text = X[:, text_slice]
        X_img = X[:, image_slice]
        X_text_and_image = self.text_and_image_fuser(torch.cat([X_text, X_img], dim=1))
        return self.out(torch.cat([X_tab, X_text_and_image], dim=1))

    @property
    def output_dim(self):
        return self.output_units


deephead = MyModelFuser(
    tab_incoming_dim=tab_mlp.output_dim,
    text_incoming_dim=models_fuser.output_dim,
    image_incoming_dim=vision.output_dim,
    output_units=8,
)

model = WideDeep(
    deeptabular=tab_mlp,
    deeptext=models_fuser,
    deepimage=vision,
    deephead=deephead,
)

trainer = Trainer(model, objective="binary")

trainer.fit(
    X_tab=X_tab,
    X_text=[X_text_1, X_text_2],
    X_img=X_img,
    target=df["target"].values,
    n_epochs=1,
    batch_size=32,
)


# 7. A Two tower model
np.random.seed(42)

# user_features dataframe
user_ids = np.arange(1, 101)
ages = np.random.randint(18, 60, size=100)
genders = np.random.choice(["male", "female"], size=100)
locations = np.random.choice(["city_a", "city_b", "city_c", "city_d"], size=100)
user_features = pd.DataFrame(
    {"id": user_ids, "age": ages, "gender": genders, "location": locations}
)

# item_features dataframe
item_ids = np.arange(1, 101)
prices = np.random.uniform(10, 500, size=100).round(2)
colors = np.random.choice(["red", "blue", "green", "black"], size=100)
categories = np.random.choice(["electronics", "clothing", "home", "toys"], size=100)

item_features = pd.DataFrame(
    {"id": item_ids, "price": prices, "color": colors, "category": categories}
)

# Interactions dataframe
interaction_user_ids = np.random.choice(user_ids, size=1000)
interaction_item_ids = np.random.choice(item_ids, size=1000)
purchased = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
interactions = pd.DataFrame(
    {
        "user_id": interaction_user_ids,
        "item_id": interaction_item_ids,
        "purchased": purchased,
    }
)
user_item_purchased = interactions.merge(
    user_features, left_on="user_id", right_on="id"
).merge(item_features, left_on="item_id", right_on="id")


# Users
tab_preprocessor_user = TabPreprocessor(
    cat_embed_cols=["gender", "location"],
    continuous_cols=["age"],
)
X_user = tab_preprocessor_user.fit_transform(user_item_purchased)
tab_mlp_user = TabMlp(
    column_idx=tab_preprocessor_user.column_idx,
    cat_embed_input=tab_preprocessor_user.cat_embed_input,
    continuous_cols=["age"],
    mlp_hidden_dims=[16, 8],
    mlp_dropout=[0.2, 0.2],
)

# Items
tab_preprocessor_item = TabPreprocessor(
    cat_embed_cols=["color", "category"],
    continuous_cols=["price"],
)
X_item = tab_preprocessor_item.fit_transform(user_item_purchased)
tab_mlp_item = TabMlp(
    column_idx=tab_preprocessor_item.column_idx,
    cat_embed_input=tab_preprocessor_item.cat_embed_input,
    continuous_cols=["price"],
    mlp_hidden_dims=[16, 8],
    mlp_dropout=[0.2, 0.2],
)

two_tower_model = ModelFuser([tab_mlp_user, tab_mlp_item], fusion_method="dot")

model = WideDeep(deeptabular=two_tower_model)

trainer = Trainer(
    model,
    objective="binary",
)

trainer.fit(
    X_tab=[X_user, X_item],
    target=interactions.purchased.values,
    n_epochs=1,
    batch_size=32,
)


# 8. Simply Tabular with a multi-target loss

# let's add a second target to the dataframe
df["target2"] = [random.choice([0, 1]) for _ in range(100)]

# Tabular
tab_preprocessor = TabPreprocessor(
    embed_cols=["city", "name"], continuous_cols=["age", "height"]
)
X_tab = tab_preprocessor.fit_transform(df)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[64, 32],
)

# 2 binary targets. For other types of targets, please, see the documentation
model = WideDeep(deeptabular=tab_mlp, pred_dim=2)

loss = MultiTargetClassificationLoss(binary_config=[0, 1], reduction="mean")

trainer = Trainer(model, objective="multitarget", custom_loss_function=loss)

trainer.fit(
    X_tab=X_tab,
    target=df[["target", "target2"]].values,
    n_epochs=1,
    batch_size=32,
)
