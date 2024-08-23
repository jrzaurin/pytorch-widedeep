import numpy as np
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep, ModelFuser
from pytorch_widedeep.preprocessing import TabPreprocessor

# set the random seed for reproducibility
np.random.seed(42)

# create the user_features table
user_ids = np.arange(1, 101)  # 100 user ids
ages = np.random.randint(18, 60, size=100)  # random ages between 18 and 60
genders = np.random.choice(["male", "female"], size=100)  # random gender assignment
locations = np.random.choice(
    ["city a", "city b", "city c", "city d"], size=100
)  # random locations

user_features = pd.DataFrame(
    {"id": user_ids, "age": ages, "gender": genders, "location": locations}
)

# create the item_features table
item_ids = np.arange(1, 101)  # 100 item ids
prices = np.random.uniform(10, 500, size=100).round(
    2
)  # random prices between $10 and $500
colors = np.random.choice(["red", "blue", "green", "black"], size=100)  # random colors
categories = np.random.choice(
    ["electronics", "clothing", "home", "toys"], size=100
)  # random categories

item_features = pd.DataFrame(
    {"id": item_ids, "price": prices, "color": colors, "category": categories}
)

# create the interactions table
interaction_user_ids = np.random.choice(
    user_ids, size=1000
)  # random user ids for interactions
interaction_item_ids = np.random.choice(
    item_ids, size=1000
)  # random item ids for interactions
purchased = np.random.choice(
    [0, 1], size=1000, p=[0.7, 0.3]
)  # 70% not purchased, 30% purchased

interactions = pd.DataFrame(
    {
        "user_id": interaction_user_ids,
        "item_id": interaction_item_ids,
        "purchased": purchased,
    }
)

user_item_purchased_df = interactions.merge(
    user_features, left_on="user_id", right_on="id"
).merge(item_features, left_on="item_id", right_on="id")


tab_preprocessor_user = TabPreprocessor(
    cat_embed_cols=["gender", "location"],
    continuous_cols=["age"],
)

tab_preprocessor_item = TabPreprocessor(
    cat_embed_cols=["color", "category"],
    continuous_cols=["price"],
)

X_user = tab_preprocessor_user.fit_transform(user_item_purchased_df)
X_item = tab_preprocessor_item.fit_transform(user_item_purchased_df)

tab_mlp_user = TabMlp(
    column_idx=tab_preprocessor_user.column_idx,
    cat_embed_input=tab_preprocessor_user.cat_embed_input,
    continuous_cols=["age"],
    mlp_hidden_dims=[16, 8],
    mlp_dropout=[0.2, 0.2],
)

tab_mlp_item = TabMlp(
    column_idx=tab_preprocessor_item.column_idx,
    cat_embed_input=tab_preprocessor_item.cat_embed_input,
    continuous_cols=["price"],
    mlp_hidden_dims=[16, 8],
    mlp_dropout=[0.2, 0.2],
)

two_tower_model = ModelFuser([tab_mlp_user, tab_mlp_item], fusion_method="dot")

# model = WideDeep(deeptabular=[tab_mlp_user, tab_mlp_item])
model = WideDeep(deeptabular=two_tower_model)

trainer = Trainer(
    model,
    objective="binary",
)

trainer.fit(
    X_tab=[X_user, X_item],
    target=interactions.purchased.values,
    batch_size=32,
    n_epochs=2,
)
