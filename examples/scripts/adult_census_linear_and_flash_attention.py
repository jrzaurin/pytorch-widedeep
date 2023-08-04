from time import time

from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, TabTransformer
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
    input_dim=16,
    n_heads=2,
    n_blocks=2,
)

linear_tab_transformer = TabTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=16,
    n_heads=2,
    n_blocks=2,
    use_linear_attention=True,
)

flash_tab_transformer = TabTransformer(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    input_dim=16,
    n_heads=2,
    n_blocks=2,
    use_flash_attention=True,
)

s_model = WideDeep(deeptabular=tab_transformer)
l_model = WideDeep(deeptabular=linear_tab_transformer)
f_model = WideDeep(deeptabular=flash_tab_transformer)

for name, model in [("standard", s_model), ("linear", l_model), ("flash", f_model)]:
    trainer = Trainer(
        model,
        objective="binary",
        metrics=[Accuracy],
    )

    s = time()
    trainer.fit(
        X_tab=X_tab_train,
        target=target,
        n_epochs=1,
        batch_size=64,
        val_split=0.2,
    )
    e = time() - s
    print(f"{name} attention time: {round(e, 3)} secs")
