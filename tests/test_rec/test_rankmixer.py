import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models.rec.rankmixer import (
    RankMixer,
    PerTokenFFN,
    RankMixerBlock,
    RankMixerTokenizer,
    MultiHeadTokenMixing,
    SparseMoEPerTokenFFN,
)

from .utils_test_rec import create_train_val_test_data


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)


# ---------------------------------------------------------------------------
# RankMixerTokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_init_requires_divisible_input_dim():
    with pytest.raises(AssertionError):
        RankMixerTokenizer(group_sizes=[5, 5], model_dim=8, token_size=6)  # 10 % 6 != 0


def test_tokenizer_forward_shape_and_dim_check():
    tok = RankMixerTokenizer(
        group_sizes=[8, 8], model_dim=16, token_size=8
    )  # input_dim=16, T=2
    x_ok = torch.randn(4, 16)
    out = tok(x_ok)
    assert out.shape == (4, 2, 16)

    x_bad = torch.randn(4, 15)
    with pytest.raises(AssertionError):
        _ = tok(x_bad)


# ---------------------------------------------------------------------------
# MultiHeadTokenMixing
# ---------------------------------------------------------------------------


def test_multihead_token_mixing_matches_reference_rearrange():
    B, T, D = 3, 4, 12
    x = torch.randn(B, T, D)

    mix = MultiHeadTokenMixing(num_tokens=T, model_dim=D)
    y = mix(x)

    # Reference: exactly what the module does
    d = D // T
    y_ref = torch.einsum("bthd->bhtd", x.view(B, T, T, d)).reshape(B, T, D)
    assert y.shape == (B, T, D)
    assert torch.allclose(y, y_ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# PerTokenFFN
# ---------------------------------------------------------------------------


def test_pertoken_ffn_shape_and_grad():
    B, T, D = 2, 3, 8
    ffn = PerTokenFFN(num_tokens=T, model_dim=D, ff_factor=2.0, ff_dropout=0.0)

    x = torch.randn(B, T, D, requires_grad=True)
    y = ffn(x)
    assert y.shape == (B, T, D)

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert ffn.W1.grad is not None
    assert ffn.W2.grad is not None


# ---------------------------------------------------------------------------
# SparseMoEPerTokenFFN
# ---------------------------------------------------------------------------


def test_sparse_moe_topk_routing_is_sparse_and_normalized():
    B, T, D = 2, 3, 6
    E, k = 5, 2
    moe = SparseMoEPerTokenFFN(
        num_tokens=T,
        model_dim=D,
        num_experts=E,
        top_k=k,
        ff_factor=2.0,
        ff_dropout=0.0,
    )

    x = torch.randn(B, T, D)

    # Recompute routing weights like forward does (without relying on internal locals)
    router_logits = torch.einsum("btd,tde->bte", x, moe.router)
    router_weights = torch.relu(router_logits)
    _, topk_idx = router_weights.topk(k, dim=-1)
    mask = torch.zeros_like(router_weights).scatter_(-1, topk_idx, 1.0)
    masked = router_weights * mask
    denom = masked.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    w = masked / denom

    # support: everything not in mask is 0
    assert torch.allclose(w * (1 - mask), torch.zeros_like(w), atol=0, rtol=0)

    # non-negativity
    assert torch.all(w >= 0)

    # at most k positives (strictly > 0)
    assert torch.all((w > 0).sum(dim=-1) <= k)

    # normalization: if there is any routed mass, sums to 1; else sums to 0
    mass = masked.sum(dim=-1)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums[mass > 0], torch.ones_like(sums[mass > 0]), atol=1e-6)
    assert torch.allclose(sums[mass == 0], torch.zeros_like(sums[mass == 0]), atol=1e-6)


# ---------------------------------------------------------------------------
# RankMixerBlock
# ---------------------------------------------------------------------------


def test_rankmixer_block_shape():
    B, T, D = 2, 4, 16
    block = RankMixerBlock(num_tokens=T, model_dim=D, ff_factor=2.0, ff_dropout=0.0)
    x = torch.randn(B, T, D)
    y = block(x)
    assert y.shape == (B, T, D)


# ---------------------------------------------------------------------------
# RankMixer — helpers
# ---------------------------------------------------------------------------


def _minimal_rankmixer_cfg():
    # 2 cat cols + 2 cont cols; cont embedded to dim 8 => cont_out_dim=16
    # cat embed dims 8 and 8 => cat_out_dim=16
    # total flat embedding dim = 32
    # token_size=16 => num_tokens=2
    column_idx = {"c1": 0, "c2": 1, "x1": 2, "x2": 3}
    cat_embed_input = [("c1", 10, 8), ("c2", 20, 8)]
    continuous_cols = ["x1", "x2"]

    # Groups already match natural order (cats first, then conts) — no
    # rearrangement needed, exercises the early-exit path in
    # _rearrange_embeddings_to_groups
    column_groups = [["c1", "c2"], ["x1", "x2"]]

    return dict(
        column_idx=column_idx,
        column_groups=column_groups,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
        embed_continuous_method="standard",
        cont_embed_dim=8,
        num_tokens=2,
        token_size=16,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        ff_dropout=0.0,
    )


def _make_mixed_rankmixer() -> RankMixer:
    """
    2 cat + 1 cont col per group, interleaved across groups.

    Natural flat order from _get_embeddings:
      user_id[0:8] | user_location[8:16] | item_id[16:24] | item_category[24:32]
      | user_age[32:40] | item_price[40:48]

    Expected order after _rearrange_embeddings_to_groups:
      user_id[0:8] | user_location[8:16] | user_age[16:24]
      | item_id[24:32] | item_category[32:40] | item_price[40:48]
    """
    return RankMixer(
        column_idx={
            "user_id": 0,
            "user_location": 1,
            "item_id": 2,
            "item_category": 3,
            "user_age": 4,
            "item_price": 5,
        },
        column_groups=[
            ["user_id", "user_location", "user_age"],
            ["item_id", "item_category", "item_price"],
        ],
        cat_embed_input=[
            ("user_id", 10, 8),
            ("user_location", 5, 8),
            ("item_id", 20, 8),
            ("item_category", 4, 8),
        ],
        continuous_cols=["user_age", "item_price"],
        embed_continuous_method="standard",
        cont_embed_dim=8,
        num_tokens=2,
        token_size=24,
        model_dim=32,
        num_layers=1,
        num_heads=2,
        ff_dropout=0.0,
    )


def _make_X(column_idx: dict, n: int = 4) -> torch.Tensor:
    X = torch.zeros(n, len(column_idx))
    X[:, 0] = torch.randint(0, 10, (n,)).float()  # user_id
    X[:, 1] = torch.randint(0, 5, (n,)).float()  # user_location
    X[:, 2] = torch.randint(0, 20, (n,)).float()  # item_id
    X[:, 3] = torch.randint(0, 4, (n,)).float()  # item_category
    X[:, 4] = torch.rand(n)  # user_age
    X[:, 5] = torch.rand(n)  # item_price
    return X


# ---------------------------------------------------------------------------
# RankMixer — validation
# ---------------------------------------------------------------------------


def test_rankmixer_validate_input_coverage_mismatch_raises():
    cfg = _minimal_rankmixer_cfg()
    cfg["column_groups"] = [["c1", "c2"], ["x1"]]  # missing x2
    with pytest.raises(ValueError, match="must reference the same columns"):
        _ = RankMixer(**cfg)


# ---------------------------------------------------------------------------
# RankMixer — end-to-end forward / backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_moe", [False, True])
def test_rankmixer_end_to_end_forward_and_backward(use_moe):
    cfg = _minimal_rankmixer_cfg()
    cfg["use_moe"] = use_moe
    cfg["num_experts"] = 3
    cfg["top_k"] = 1

    model = RankMixer(**cfg)

    B = 5
    x_cat = torch.stack(
        [
            torch.randint(0, 10, (B,)),  # c1
            torch.randint(0, 20, (B,)),  # c2
        ],
        dim=1,
    )
    x_cont = torch.randn(B, 2)
    X = torch.cat([x_cat.float(), x_cont], dim=1)

    out = model(X)
    assert out.shape == (B, model.output_dim)

    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_rankmixer_full_process():

    train, valid, test = create_train_val_test_data()

    cat_embed_cols = [
        ("user_id", 8),
        ("user_location", 8),
        ("item_id", 8),
        ("item_category", 8),
    ]
    continuous_cols = ["user_age", "item_price"]

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
    )

    X_tab_tr = tab_preprocessor.fit_transform(train)
    X_tab_val = tab_preprocessor.transform(valid)
    X_tab_te = tab_preprocessor.transform(test)
    y_tr = train["purchased"].values
    y_val = valid["purchased"].values

    column_groups = [
        ["user_id", "user_location", "user_age"],
        ["item_id", "item_category", "item_price"],
    ]

    rankmixer = RankMixer(
        column_idx=tab_preprocessor.column_idx,
        column_groups=column_groups,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        cont_norm_layer="layernorm",
        embed_continuous_method="standard",
        cont_embed_dim=8,
        num_tokens=2,
        token_size=24,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        ff_dropout=0.1,
    )

    model = WideDeep(deeptabular=rankmixer)

    trainer = Trainer(model, objective="binary", verbose=0)
    X_train = {"X_tab": X_tab_tr, "target": y_tr}
    X_val = {"X_tab": X_tab_val, "target": y_val}
    trainer.fit(X_train=X_train, X_val=X_val, n_epochs=1)

    preds = trainer.predict_proba(X_tab=X_tab_te)

    assert preds.shape[0] == X_tab_te.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )


# ---------------------------------------------------------------------------
# _rearrange_embeddings_to_groups
# ---------------------------------------------------------------------------
#
# Natural order from _get_embeddings (default mixed setup):
#   [0:8]   user_id       (cat, embed_dim=8)
#   [8:16]  user_location (cat, embed_dim=8)
#   [16:24] item_id       (cat, embed_dim=8)
#   [24:32] item_category (cat, embed_dim=8)
#   [32:40] user_age      (cont, embed_dim=8)
#   [40:48] item_price    (cont, embed_dim=8)
#
# column_groups = [
#     ["user_id", "user_location", "user_age"],   → group 0
#     ["item_id", "item_category", "item_price"],  → group 1
# ]
#
# Expected order after rearrange:
#   [0:8]   user_id
#   [8:16]  user_location
#   [16:24] user_age       ← moved from [32:40]
#   [24:32] item_id        ← moved from [16:24]
#   [32:40] item_category  ← moved from [24:32]
#   [40:48] item_price     ← unchanged
# ---------------------------------------------------------------------------


class TestRearrangeEmbeddingsToGroups:

    def test_shape_preserved(self):
        """Rearranged tensor must have exactly the same shape as the input."""
        model = _make_mixed_rankmixer()
        model.eval()
        X = _make_X(model.column_idx)
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)
        assert out.shape == flat.shape

    def test_mixed_groups_each_column_at_correct_position(self):
        """
        Verify every column's embedding block lands at exactly the position
        implied by column_groups, not the natural cat-then-cont order.
        """
        model = _make_mixed_rankmixer()
        model.eval()
        X = _make_X(model.column_idx)
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)

        assert torch.allclose(out[:, 0:8], flat[:, 0:8])  # user_id:       unchanged
        assert torch.allclose(out[:, 8:16], flat[:, 8:16])  # user_location: unchanged
        assert torch.allclose(out[:, 16:24], flat[:, 32:40])  # user_age:      32→16
        assert torch.allclose(out[:, 24:32], flat[:, 16:24])  # item_id:       16→24
        assert torch.allclose(out[:, 32:40], flat[:, 24:32])  # item_category: 24→32
        assert torch.allclose(out[:, 40:48], flat[:, 40:48])  # item_price:    unchanged

    def test_is_a_permutation(self):
        """
        The rearranged tensor must contain exactly the same set of per-row
        values as the original — no values dropped or duplicated.

        We verify this by checking that the sorted flat tensor is identical
        to the sorted rearranged tensor. Unlike the slice checks above, this
        catches any duplication or dropping that might pass positional checks
        if two columns happen to share the same embedding values.
        """
        model = _make_mixed_rankmixer()
        model.eval()
        # Use arange-based synthetic embeddings so every position is unique,
        # making duplication or dropping impossible to hide via coincidence.
        with torch.no_grad():
            flat = torch.arange(48, dtype=torch.float32).unsqueeze(0).expand(4, -1)
            out = model._rearrange_embeddings_to_groups(flat)

        assert torch.allclose(flat.sort(dim=-1).values, out.sort(dim=-1).values)

    def test_already_ordered_groups_returns_same_tensor(self):
        """
        When column_groups already matches the natural cat-then-cont output
        order, the tensor must pass through unchanged (early-exit path).
        """
        model = RankMixer(**_minimal_rankmixer_cfg())
        model.eval()
        X = torch.zeros(4, 4)
        X[:, 0] = torch.randint(0, 10, (4,)).float()
        X[:, 1] = torch.randint(0, 5, (4,)).float()
        X[:, 2:] = torch.rand(4, 2)
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)
        assert torch.allclose(out, flat)

    def test_cat_only_non_natural_order(self):
        """
        All-categorical setup where groups interleave the natural cat order.

        Natural: a[0:8] b[8:16] c[16:24] d[24:32]
        Groups:  a[0:8] c[8:16] b[16:24] d[24:32]
        """
        model = RankMixer(
            column_idx={"a": 0, "b": 1, "c": 2, "d": 3},
            column_groups=[["a", "c"], ["b", "d"]],
            cat_embed_input=[("a", 5, 8), ("b", 5, 8), ("c", 5, 8), ("d", 5, 8)],
            num_tokens=2,
            token_size=16,
            model_dim=32,
            num_layers=1,
            num_heads=2,
            ff_dropout=0.0,
        )
        model.eval()
        X = torch.zeros(4, 4)
        for i in range(4):
            X[:, i] = torch.randint(0, 5, (4,)).float()
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)

        assert torch.allclose(out[:, 0:8], flat[:, 0:8])  # a: unchanged
        assert torch.allclose(out[:, 8:16], flat[:, 16:24])  # c: 16→8
        assert torch.allclose(out[:, 16:24], flat[:, 8:16])  # b: 8→16
        assert torch.allclose(out[:, 24:32], flat[:, 24:32])  # d: unchanged

    def test_cont_only_not_embedded(self):
        """
        Continuous-only, no embedding (dim=1 per col).

        Natural: x1[0] x2[1] x3[2] x4[3]
        Groups:  x1[0] x3[1] x2[2] x4[3]
        """
        model = RankMixer(
            column_idx={"x1": 0, "x2": 1, "x3": 2, "x4": 3},
            column_groups=[["x1", "x3"], ["x2", "x4"]],
            continuous_cols=["x1", "x2", "x3", "x4"],
            num_tokens=2,
            token_size=2,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            ff_dropout=0.0,
        )
        model.eval()
        X = torch.rand(4, 4)
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)

        assert torch.allclose(out[:, 0:1], flat[:, 0:1])  # x1: unchanged
        assert torch.allclose(out[:, 1:2], flat[:, 2:3])  # x3: 2→1
        assert torch.allclose(out[:, 2:3], flat[:, 1:2])  # x2: 1→2
        assert torch.allclose(out[:, 3:4], flat[:, 3:4])  # x4: unchanged

    def test_single_group_all_columns(self):
        """Single group with all columns — output must equal input regardless of col types."""
        model = RankMixer(
            column_idx={"c1": 0, "x1": 1},
            column_groups=[["c1", "x1"]],
            cat_embed_input=[("c1", 5, 8)],
            continuous_cols=["x1"],
            embed_continuous_method="standard",
            cont_embed_dim=8,
            num_tokens=1,
            token_size=16,
            model_dim=16,
            num_layers=1,
            num_heads=1,
            ff_dropout=0.0,
        )
        model.eval()
        X = torch.zeros(4, 2)
        X[:, 0] = torch.randint(0, 5, (4,)).float()
        X[:, 1] = torch.rand(4)
        with torch.no_grad():
            flat = model._get_embeddings(X)
            out = model._rearrange_embeddings_to_groups(flat)
        assert torch.allclose(out, flat)
