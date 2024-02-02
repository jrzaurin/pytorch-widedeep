import pandas as pd

from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing.tab_preprocessor import TabPreprocessor

df: pd.DataFrame = load_adult(as_frame=True)
df.columns = [c.replace("-", "_") for c in df.columns]

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
quantisation_setup = {"age": 5, "hours_per_week": 5}

tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols,
    continuous_cols=continuous_cols,
    quantization_setup=quantisation_setup,
)

X_tab = tab_preprocessor.fit_transform(df)
df_decoded = tab_preprocessor.inverse_transform(X_tab)
