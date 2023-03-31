import torch
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import (
    SAINT,
    WideDeep,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
)
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor

# use_cuda = torch.cuda.is_available()

df = load_adult(as_frame=True)
df.columns = [c.replace("-", "_") for c in df.columns]
df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop("income", axis=1, inplace=True)
target = "income_label"

cat_embed_cols = []
for col in df.columns:
    if df[col].dtype == "O" or df[col].nunique() < 200 and col != target:
        cat_embed_cols.append(col)

train, test = train_test_split(df, test_size=0.1, random_state=1, stratify=df[[target]])

tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_embed_cols, with_attention=True)

X_tab_train = tab_preprocessor.fit_transform(train)
X_tab_test = tab_preprocessor.transform(test)
target = train[target].values

tab_transformer = TabTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=2,
)

saint = SAINT(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=1,
)

tab_fastformer = TabFastFormer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=1,
)

ft_transformer = FTTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=2,
    kv_compression_factor=1.0,
)


model = WideDeep(deeptabular=tab_transformer)

trainer = Trainer(
    model,
    objective="binary",
    metrics=[Accuracy],
)

trainer.fit(
    X_tab=X_tab_train,
    target=target,
    n_epochs=5,
    batch_size=128,
    val_split=0.2,
)


def feature_importance_tab_transformer(trainer, X_tab_test):
    _ = trainer.predict(X_tab=X_tab_test[:32])

    attention_weights = trainer.model.deeptabular[0].attention_weights

    if trainer.model.deeptabular[0].with_cls_token:
        feat_imp = torch.stack(
            [aw.mean(1)[:, 0, 1:] for aw in attention_weights], dim=0
        ).mean(0)
    else:
        feat_imp = torch.stack([aw.mean(1).mean(1) for aw in attention_weights]).mean(0)

    return feat_imp


feat_imp = feature_importance_tab_transformer(trainer, X_tab_test)

assert feat_imp.shape[1] == len(cat_embed_cols)


# also this is valid
def _feature_importance_tab_transformer(trainer, X_tab_test):
    _ = trainer.predict(X_tab=X_tab_test[:32])

    attention_weights = torch.stack(
        trainer.model.deeptabular[0].attention_weights, dim=0
    )

    if trainer.model.deeptabular[0].with_cls_token:
        feat_imp = attention_weights.mean(0).mean(1)[:, 0, 1:]
    else:
        feat_imp = attention_weights.mean(0).mean(1).mean(1)

    return feat_imp


feat_imp = _feature_importance_tab_transformer(trainer, X_tab_test)

assert feat_imp.shape[1] == len(cat_embed_cols)


# THESE TWO ARE WRONG
# def get_feature_importance_saint(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     last_block_col_attention_weights = trainer.model.deeptabular[0].attention_weights[
#         -1
#     ][0]

#     if trainer.model.deeptabular[0].with_cls_token:
#         feat_imp = last_block_col_attention_weights[..., 0].mean(1)[:, 1:]
#     else:
#         feat_imp = last_block_col_attention_weights.mean(1).mean(1)

#     return feat_imp


# def get_feature_importance_fastformer(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     last_block_attention_weights = trainer.model.deeptabular[0].attention_weights[-1]

#     if trainer.model.deeptabular[0].with_cls_token:
#         feat_imp = torch.stack(
#             [attn_weight.mean(1) for attn_weight in last_block_attention_weights]
#         ).mean(0)[:, 1:]
#     else:
#         feat_imp = torch.stack(
#             [attn_weight.mean(1) for attn_weight in last_block_attention_weights]
#         ).mean(0)

#     return feat_imp


# def get_feature_importance_ft_transformer(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     attention_weights = torch.cat(trainer.model.deeptabular[0].attention_weights)

#     if trainer.model.deeptabular[0].with_cls_token:
#         # I need to check whether the .mean(1).mean(1) is .mean(1).mean(0)
#         feat_imp = attention_weights.mean(1).mean(1).reshape(32, 2, 14).mean(1)[:, 1:]
#     else:
#         feat_imp = attention_weights.mean(1).mean(1).reshape(32, 2, 13).mean(1)

#     return feat_imp
