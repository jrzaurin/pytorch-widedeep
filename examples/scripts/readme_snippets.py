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
from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    Vision,
    BasicRNN,
    WideDeep,
    ModelFuser,
)
from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    TextPreprocessor,
    WidePreprocessor,
    ImagePreprocessor,
)
from pytorch_widedeep.models._base_wd_model_component import (
    BaseWDModelComponent,
)


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
