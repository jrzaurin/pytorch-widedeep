from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import (
    SAINT,
    TabNet,
    WideDeep,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
    SelfAttentionMLP,
    ContextAttentionMLP,
)
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor

# use_cuda = torch.cuda.is_available()

df = load_adult(as_frame=True)
df.columns = [c.replace("-", "_") for c in df.columns]
df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop("income", axis=1, inplace=True)
target_colname = "income_label"

cat_embed_cols = []
for col in df.columns:
    if df[col].dtype == "O" or df[col].nunique() < 200 and col != target_colname:
        cat_embed_cols.append(col)

train, test = train_test_split(
    df, test_size=0.1, random_state=1, stratify=df[[target_colname]]
)

with_cls_token = True
tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols, with_attention=True, with_cls_token=with_cls_token
)

X_tab_train = tab_preprocessor.fit_transform(train)
X_tab_test = tab_preprocessor.transform(test)
target = train[target_colname].values


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
    n_blocks=2,
)

tab_fastformer = TabFastFormer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=2,
)

ft_transformer = FTTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=8,
    n_heads=2,
    n_blocks=2,
    kv_compression_factor=1.0,  # if this is diff than one, we cannot do this
)

context_attention_mlp = ContextAttentionMLP(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=16,
    attn_dropout=0.2,
    n_blocks=3,
)

self_attention_mlp = SelfAttentionMLP(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    n_blocks=3,
)

for attention_based_model in [
    tab_transformer,
    saint,
    tab_fastformer,
    ft_transformer,
    context_attention_mlp,
    self_attention_mlp,
]:
    model = WideDeep(deeptabular=attention_based_model)  # type: ignore[arg-type]

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
        feature_importance_sample_size=1000,
    )

    assert (
        len(trainer.feature_importance) == X_tab_train.shape[1] - 1
        if with_cls_token
        else X_tab_train.shape[1]
    )

train, test = train_test_split(
    df, test_size=0.1, random_state=1, stratify=df[[target_colname]]
)

tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_embed_cols)

X_tab_train = tab_preprocessor.fit_transform(train)
X_tab_test = tab_preprocessor.transform(test)
target = train[target_colname].values

tabnet = TabNet(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
)

model = WideDeep(deeptabular=tabnet)

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
    feature_importance_sample_size=1000,
)

assert len(trainer.feature_importance) == X_tab_train.shape[1]

# def feature_importance_tab_transformer(trainer, X_tab_test):
#     attention_weights_1 = trainer.model.deeptabular[0].attention_weights

#     attention_weights_2 = torch.stack(
#         trainer.model.deeptabular[0].attention_weights, dim=0
#     )

#     if trainer.model.deeptabular[0].with_cls_token:
#         # mean through heads -> get the cls token attention weights -> get the
#         # attention weights for all features BUT cls token. Then do this for
#         # all attention blocks and compute the mean
#         feat_imp_1 = torch.stack(
#             [aw.mean(1)[:, 0, 1:] for aw in attention_weights_1], dim=0
#         ).mean(0)
#         # Also valid
#         feat_imp_2 = attention_weights_2.mean(0).mean(1)[:, 0, 1:]

#         assert torch.allclose(feat_imp_1, feat_imp_2)
#     else:
#         # mean through heads -> mean through dim 1 (ith element of each row)
#         # as these weight represent how much a given feature is attented to.
#         # Then do this for all attention blocks and compute the mean
#         feat_imp_1 = torch.stack(
#             [aw.mean(1).mean(1) for aw in attention_weights_1]
#         ).mean(0)
#         # Also valid
#         feat_imp_2 = attention_weights_2.mean(0).mean(1).mean(1)

#         assert torch.allclose(feat_imp_1, feat_imp_2)

#     return feat_imp_1

# def get_feature_importance_saint(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     attention_weights = torch.stack(
#         [aw[0] for aw in trainer.model.deeptabular[0].attention_weights], dim=0
#     )

#     if trainer.model.deeptabular[0].with_cls_token:
#         feat_imp = attention_weights.mean(0).mean(1)[:, 0, 1:]
#     else:
#         feat_imp = attention_weights.mean(0).mean(1).mean(1)

#     return feat_imp


# def get_feature_importance_fastformer(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     # To compute feature importance we could:
#     # 1. Use only beta attention weights since one could say that they have implicit information about the alpha weights
#     # 2. Use the mean of alpha and beta attention weights

#     # Approach 2
#     alpha_weights, beta_weights = zip(*trainer.model.deeptabular[0].attention_weights)
#     attention_weights = torch.stack(alpha_weights + beta_weights, dim=0)

#     if trainer.model.deeptabular[0].with_cls_token:
#         # mean through blocks -> mean through heads -> all but cls token column
#         feat_imp = attention_weights.mean(0).mean(1)[:, 1:]
#     else:
#         # mean through blocks -> mean through heads
#         feat_imp = attention_weights.mean(0).mean(1)

#     return feat_imp

# def get_feature_importance_ft_transformer(trainer, X_tab_test):
#     _ = trainer.predict(X_tab=X_tab_test[:32])

#     attention_weights = torch.stack(
#         trainer.model.deeptabular[0].attention_weights, dim=0
#     )

#     if trainer.model.deeptabular[0].with_cls_token:
#         feat_imp = attention_weights.mean(0).mean(1)[:, 0, 1:]
#     else:
#         feat_imp = attention_weights.mean(0).mean(1).mean(1)

#     return feat_imp
