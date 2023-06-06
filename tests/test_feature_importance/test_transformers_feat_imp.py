import numpy as np
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, TabTransformer
from pytorch_widedeep.preprocessing import TabPreprocessor

np.random.seed(42)

# Define the column names
cat_cols = ["cat1", "cat2", "cat3", "cat4"]
cont_cols = ["cont1", "cont2", "cont3", "cont4"]
columns = cat_cols + cont_cols

# Generate random categorical data
categorical_data = np.random.choice(["A", "B", "C"], size=(32, 4))

# Generate random numerical data
numerical_data = np.random.randn(32, 4)

# Create the DataFrame
data = np.concatenate((categorical_data, numerical_data), axis=1)
df = pd.DataFrame(data, columns=columns)
target = np.random.choice(2, 32)

df_tr = df[:16].copy()
df_te = df[16:].copy().reset_index(drop=True)

y_tr = target[:16]
y_te = target[16:]

# ############################TESTS BEGIN #######################################

with_cls_token = False
tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_cols,
    continuous_cols=cont_cols,
    with_attention=True,
    with_cls_token=with_cls_token,
)
X_tr = tab_preprocessor.fit_transform(df_tr).astype(float)
X_te = tab_preprocessor.transform(df_te).astype(float)

tab_transformer = TabTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    embed_continuous=True,
    input_dim=6,
    n_heads=2,
    n_blocks=2,
)

model = WideDeep(deeptabular=tab_transformer)

trainer = Trainer(
    model,
    objective="binary",
)

trainer.fit(
    X_tab=X_tr,
    target=target,
    n_epochs=1,
    batch_size=16,
)

feat_imps = trainer.feature_importance
feat_imp_per_sample = trainer.explain(X_te)
