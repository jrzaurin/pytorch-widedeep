import torch
import pytest

from pytorch_widedeep.models import TabMlp, HFModel, WideDeep
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.preprocessing import HFPreprocessor as HFTokenizer
from pytorch_widedeep.preprocessing import TabPreprocessor

from .generate_fake_data import generate

df = generate()

model_names = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "FacebookAI/roberta-base",
    "albert-base-v2",
    "google/electra-base-discriminator",
]


@pytest.mark.parametrize("model_name", model_names)
def test_model_basic_usage(model_name):
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name=model_name)
        X_text = tokenizer.encode(df.random_sentences.tolist())
        model = HFModel(model_name=model_name)
        X_text_tnsr = torch.tensor(X_text)
        out = model(X_text_tnsr)
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == model.output_dim


@pytest.mark.parametrize("model_name", model_names)
def test_model_with_params(model_name):
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name=model_name)
        X_text = tokenizer.encode(df.random_sentences.tolist())
        model = HFModel(
            model_name=model_name,
            output_attentions=True,
        )
        X_text_tnsr = torch.tensor(X_text)
        out = model(X_text_tnsr)

    attn_shape_assertion = model.attention_weight[0].shape == (
        df.shape[0],
        (
            model.config.n_heads
            if hasattr(model.config, "n_heads")
            else model.config.num_attention_heads
        ),
        X_text.shape[1],
        X_text.shape[1],
    )
    attn_len_assertion = True  # just to avoid an unbound type warning
    if model_name == "distilbert-base-uncased":
        attn_len_assertion = len(model.attention_weight) == model.config.n_layers

    if model_name == "bert-base-uncased":
        attn_len_assertion = (
            len(model.attention_weight) == model.config.num_hidden_layers
        )

    if model_name == "FacebookAI/roberta-base":
        attn_len_assertion = (
            len(model.attention_weight) == model.config.num_hidden_layers
        )

    if model_name == "albert-base-v2":
        attn_len_assertion = (
            len(model.attention_weight) == model.config.num_hidden_layers
        )

    if model_name == "google/electra-base-discriminator":
        attn_len_assertion = (
            len(model.attention_weight) == model.config.num_hidden_layers
        )

    assert attn_len_assertion and attn_shape_assertion
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == model.output_dim


def test_model_with_head(model_name="distilbert-base-uncased"):
    with pytest.warns(UserWarning):
        tokenizer = HFTokenizer(model_name=model_name)
        X_text = tokenizer.encode(df.random_sentences.tolist())
        model = HFModel(
            model_name=model_name,
            use_cls_token=False,
            head_hidden_dims=[64, 32],
        )
        X_text_tnsr = torch.tensor(X_text)
        out = model(X_text_tnsr)
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == 32


def test_hf_model_combined_with_tabmlp(model_name="distilbert-base-uncased"):
    with pytest.warns(UserWarning):
        tab_preprocessor = TabPreprocessor(
            embed_cols=["cat1", "cat2"], continuous_cols=["num1", "num2"]
        )
        X_tab = tab_preprocessor.fit_transform(df)
        tabmlp = TabMlp(
            column_idx=tab_preprocessor.column_idx,
            cat_embed_input=tab_preprocessor.cat_embed_input,
            continuous_cols=tab_preprocessor.continuous_cols,
            cont_norm_layer="batchnorm",
            mlp_hidden_dims=[32, 16],
            mlp_activation="relu",
            mlp_dropout=0.1,
        )

        tokenizer = HFTokenizer(model_name=model_name)
        X_text = tokenizer.encode(df.random_sentences.tolist())
        hf_model = HFModel(
            model_name=model_name,
            use_cls_token=True,
            head_hidden_dims=[64, 32],
        )
        wide_deep = WideDeep(deeptabular=tabmlp, deeptext=hf_model)

        out = wide_deep(
            {"deeptabular": torch.tensor(X_tab), "deeptext": torch.tensor(X_text)},
        )

    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == 1


@pytest.mark.parametrize("model_name", model_names)
def test_full_training_process(model_name):
    tab_preprocessor = TabPreprocessor(
        embed_cols=["cat1", "cat2"], continuous_cols=["num1", "num2"]
    )
    X_tab = tab_preprocessor.fit_transform(df)
    tabmlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        cont_norm_layer="batchnorm",
        mlp_hidden_dims=[32, 16],
        mlp_activation="relu",
        mlp_dropout=0.1,
    )

    tokenizer = HFTokenizer(
        model_name=model_name,
        text_col="random_sentences",
    )
    X_text = tokenizer.fit_transform(df)
    hf_model = HFModel(
        model_name=model_name,
        use_cls_token=True,
    )
    model = WideDeep(deeptabular=tabmlp, deeptext=hf_model)

    trainer = Trainer(
        model,
        objective="binary",
        verbose=0,
    )

    trainer.fit(
        X_tab=X_tab,
        X_text=X_text,
        target=df.target.values,
        batch_size=8,
    )

    preds = trainer.predict(
        X_tab=X_tab[:2],
        X_text=X_text[:2],
    )

    assert len(trainer.history) > 0 and "train_loss" in trainer.history.keys()
    assert preds.shape[0] == 2
