import string

import numpy as np
import torch

from pytorch_widedeep.models.tab_transformer import *  # noqa: F403

# I am going over test this model due to the number of components

n_embed = 5
#  this is the number of embed_cols and cont_cols. So total num of cols =
#  n_cols * 2
n_cols = 2
batch_size = 10
colnames = list(string.ascii_lowercase)[: (n_cols * 2)]
embed_cols = [np.random.choice(np.arange(n_embed), batch_size) for _ in range(n_cols)]
cont_cols = [np.random.rand(batch_size) for _ in range(n_cols)]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_emb = X_tab[:, :n_cols]
X_tab_cont = X_tab[:, n_cols:]

###############################################################################
# Test functioning using the defaults
###############################################################################

embed_input = [(u, i) for u, i in zip(colnames[:2], [n_embed] * 2)]
model1 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
)


def test_embeddings_have_padding():
    res = []
    for k, v in model1.embed_layers.items():
        res.append(v.weight.size(0) == n_embed + 1)
        res.append(not torch.all(v.weight[0].bool()))
    assert all(res)


def test_tabtransformer_output():
    out = model1(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32 + len(cont_cols)) * 2


###############################################################################
# Test SharedEmbeddings
###############################################################################

# all manually passed
def test_tabtransformer_shared_embeddings():

    res = []

    shared_embeddings = SharedEmbeddings(
        num_embed=5,
        embed_dim=16,
        embed_dropout=0.0,
        add_shared_embed=False,
        frac_shared_embed=8,
    )

    X_inp = X_tab[:, 0]
    se = shared_embeddings(X_inp.long())

    res.append((se[:, :2][0] == se[:, :2]).all())

    shared_embeddings = SharedEmbeddings(
        num_embed=5,
        embed_dim=16,
        embed_dropout=0.0,
        add_shared_embed=True,
        frac_shared_embed=8,
    )

    X_inp = X_tab[:, 0]
    se = shared_embeddings(X_inp.long())
    not_se = shared_embeddings.embed(X_inp.long())
    res.append((not_se + shared_embeddings.shared_embed).allclose(se))

    assert all(res)


model2 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
    shared_embed=True,
)


def test_shared_embeddings_have_padding():
    res = []
    for k, v in model2.embed_layers.items():
        res.append(v.embed.weight.size(0) == n_embed + 1)
        res.append(not torch.all(v.embed.weight[0].bool()))
    assert all(res)


def test_tabtransformer_w_shared_emb_output():
    out = model2(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32 + len(cont_cols)) * 2


###############################################################################
# Sanity Check: Test w/o continuous features
###############################################################################

model3 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=None,
)


def test_tabtransformer_output_no_cont():
    out = model3(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32) * 2


###############################################################################
# Test keep attention weights
###############################################################################

model4 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=None,
    keep_attn_weights=True,
)


def test_tabtransformer_keep_attn():
    out = model4(X_tab)
    assert (
        out.size(0) == 10
        and out.size(1) == (n_cols * 32) * 2
        and len(model4.attention_weights) == model4.num_blocks
        and list(model4.attention_weights[0].shape)
        == [10, model4.num_heads, n_cols, n_cols]
    )


###############################################################################
# Test full embed dropout
###############################################################################


def test_full_embed_dropout():
    bsz = 1
    cat = 10
    esz = 4
    full_embedding_dropout = FullEmbeddingDropout(dropout=0.5)
    inp = torch.rand(bsz, cat, esz)
    out = full_embedding_dropout(inp)
    # simply check that at least 1 full row is all 0s
    assert torch.any(torch.sum(out[0] == 0, axis=1) == esz)


###############################################################################
# Test fixed_attention
###############################################################################


def test_fixed_attention():
    bsz = 1
    cat = 10
    esz = 32
    multi_head_dattention = MultiHeadedAttention(
        input_dim=esz,
        num_heads=4,
        keep_attn_weights=False,
        dropout=0.1,
        fixed_attention=True,
        num_cat_columns=cat,
    )
    inp = torch.rand(bsz, cat, esz)
    try:
        out = multi_head_dattention(inp)  # noqa: F841
        if out.size() == inp.size():
            res = True
        else:
            res = False
    except Exception:
        res = False
    assert res
