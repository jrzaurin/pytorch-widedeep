import numpy as np
import torch
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.utils.deeptabular_utils import (
    LabelEncoder,
    find_bin,
    get_kernel_window,
)
from pytorch_widedeep.preprocessing.tab_preprocessor import embed_sz_rule


def create_test_dataset(input_type, input_type_2=None):
    df = pd.DataFrame()
    col1 = list(np.random.choice(input_type, 3))
    if input_type_2 is not None:
        col2 = list(np.random.choice(input_type_2, 3))
    else:
        col2 = list(np.random.choice(input_type, 3))
    df["col1"], df["col2"] = col1, col2
    return df


some_letters = ["a", "b", "c", "d", "e"]
some_numbers = [1, 2, 3, 4, 5]

df_letters = create_test_dataset(some_letters)
df_numbers = create_test_dataset(some_numbers)


###############################################################################
# Simple test of functionality: testing the LabelEncoder class
###############################################################################
le_letters = LabelEncoder(["col1", "col2"])
df_letters_le = le_letters.fit_transform(df_letters)
le_numbers = LabelEncoder(["col1", "col2"])
df_numbers_le = le_numbers.fit_transform(df_numbers)


@pytest.mark.parametrize(
    "input_df, encoder, output_df",
    [(df_letters, le_letters, df_letters_le), (df_numbers, le_numbers, df_numbers_le)],
)
def test_label_encoder(input_df, encoder, output_df):
    original_df = encoder.inverse_transform(output_df)
    assert original_df.equals(input_df)


################################################################################
# Test the TabPreprocessor: only categorical columns to be represented with
# embeddings
###############################################################################
cat_embed_cols = [("col1", 5), ("col2", 5)]

preprocessor1 = TabPreprocessor(cat_embed_cols)  # type: ignore[arg-type]
X_letters = preprocessor1.fit_transform(df_letters)

preprocessor2 = TabPreprocessor(cat_embed_cols)  # type: ignore[arg-type]
X_numbers = preprocessor2.fit_transform(df_numbers)

error_list = []


@pytest.mark.parametrize(
    "input_df, X_deep, preprocessor",
    [(df_letters, X_letters, preprocessor1), (df_numbers, X_numbers, preprocessor2)],
)
def test_prepare_deep_without_continous_columns(input_df, X_deep, preprocessor):
    for i, c in enumerate(input_df.columns):
        if (
            # remember we have an "unseen class"
            input_df[c].nunique() != preprocessor.cat_embed_input[i][1]
            or cat_embed_cols[i][1] != preprocessor.cat_embed_input[i][2]
        ):
            error_list.append(
                "error: the setup output does not match the intended input"
            )

    tmp_df = preprocessor.label_encoder.inverse_transform(
        pd.DataFrame({"col1": X_deep[:, 0], "col2": X_deep[:, 1]})
    )

    if not tmp_df.equals(input_df):
        error_list.append("error: the decoding does not match the encoding")

    assert not error_list, "errors occured:\n{}".format("\n".join(error_list))


################################################################################
# Test the TabPreprocessor: only continouos columns
###############################################################################
def test_prepare_deep_without_embedding_columns():

    errors = []
    df_randint = pd.DataFrame(np.random.choice(np.arange(100), (100, 2)))
    df_randint.columns = ["col1", "col2"]
    preprocessor3 = TabPreprocessor(continuous_cols=["col1", "col2"])

    try:
        X_randint = preprocessor3.fit_transform(df_randint)
    except Exception:
        errors.append("Fundamental Error")

    out_booleans = []

    means, stds = np.mean(X_randint, axis=0), np.std(X_randint, axis=0)
    for mean, std in zip(means, stds):
        out_booleans.append(np.isclose(mean, 0.0))
        out_booleans.append(np.isclose(std, 1.0))

    if not np.all(out_booleans):
        errors.append("There is something going on with the scaler")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))


################################################################################
# Test TabPreprocessor inverse_transform
###############################################################################

df = pd.DataFrame(
    {
        "col1": ["a", "b", "c"],
        "col2": ["c", "d", "e"],
        "col3": [10, 20, 30],
        "col4": [2, 7, 9],
    }
)


@pytest.mark.parametrize(
    "embed_cols, continuous_cols, scale",
    [
        (["col1", "col2"], None, False),
        (None, ["col3", "col4"], True),
        (["col1", "col2"], ["col3", "col4"], False),
        (["col1", "col2"], ["col3", "col4"], True),
        ([("col1", 5), ("col2", 5)], ["col3", "col4"], True),
    ],
)
def test_tab_preprocessor_inverse_transform(embed_cols, continuous_cols, scale):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols,
        continuous_cols=continuous_cols,
        scale=scale,
        verbose=False,
    )
    encoded = tab_preprocessor.fit_transform(df)
    decoded = tab_preprocessor.inverse_transform(encoded)
    try:
        if isinstance(embed_cols[0], tuple):
            embed_cols = [c[0] for c in embed_cols]
        emb_df = df[embed_cols]
    except Exception:
        emb_df = pd.DataFrame()
    try:
        cont_df = df[continuous_cols]
    except Exception:
        cont_df = pd.DataFrame()
    org_df = pd.concat([emb_df, cont_df], axis=1)
    decoded = decoded.astype(org_df.dtypes.to_dict())
    assert decoded.equals(org_df)


################################################################################
# Test TabPreprocessor for the TabTransformer
###############################################################################


@pytest.mark.parametrize(
    "embed_cols, continuous_cols, scale, with_cls_token",
    [
        (["col1", "col2"], None, False, True),
        (["col1", "col2"], ["col3", "col4"], False, True),
        (["col1", "col2"], ["col3", "col4"], True, True),
        (["col1", "col2"], None, False, False),
        (["col1", "col2"], ["col3", "col4"], False, False),
        (["col1", "col2"], ["col3", "col4"], True, False),
    ],
)
def test_tab_preprocessor_trasformer(
    embed_cols, continuous_cols, scale, with_cls_token
):
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=embed_cols,
        continuous_cols=continuous_cols,
        scale=scale,
        for_transformer=True,
        with_cls_token=with_cls_token,
        verbose=False,
    )
    encoded = tab_preprocessor.fit_transform(df)
    decoded = tab_preprocessor.inverse_transform(encoded)
    try:
        if isinstance(embed_cols[0], tuple):
            embed_cols = [c[0] for c in embed_cols]
        emb_df = df[embed_cols]
    except Exception:
        emb_df = pd.DataFrame()
    try:
        cont_df = df[continuous_cols]
    except Exception:
        cont_df = pd.DataFrame()
    org_df = pd.concat([emb_df, cont_df], axis=1)
    decoded = decoded.astype(org_df.dtypes.to_dict())
    assert decoded.equals(org_df)


@pytest.mark.parametrize(
    "embed_cols, continuous_cols, scale",
    [
        (None, ["col3", "col4"], True),
        ([("col1", 5), ("col2", 5)], ["col3", "col4"], True),
    ],
)
def test_tab_preprocessor_trasformer_raise_error(embed_cols, continuous_cols, scale):
    with pytest.raises(ValueError):
        tab_preprocessor = TabPreprocessor(  # noqa: F841
            cat_embed_cols=embed_cols,
            continuous_cols=continuous_cols,
            scale=scale,
            for_transformer=True,
        )


@pytest.mark.parametrize(
    "shared_embed",
    [True, False],
)
def test_with_and_without_shared_embeddings(shared_embed):

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=["col1", "col2"],
        continuous_cols=None,
        for_transformer=True,
        shared_embed=shared_embed,
        verbose=False,
    )

    encoded = tab_preprocessor.fit_transform(df)  # noqa: F841

    first_index = []
    for k, v in tab_preprocessor.label_encoder.encoding_dict.items():
        first_index.append(min(v.values()))
        added_idx = len(v) if not shared_embed else 0

    if shared_embed:
        res = len(set(first_index)) == 1
    else:
        res = (
            len(set(first_index)) == 2 and first_index[1] == first_index[0] + added_idx
        )
    assert res


###############################################################################
# Test NotFittedError
###############################################################################


def test_notfittederror():
    processor = TabPreprocessor(
        cat_embed_cols=["col1", "col2"], continuous_cols=["col3", "col4"]
    )
    with pytest.raises(NotFittedError):
        processor.transform(df)


###############################################################################
# Test embeddings fastai's rule of thumb
###############################################################################


@pytest.mark.parametrize(
    "rule",
    [
        ("google"),
        ("fastai_old"),
        ("fastai_new"),
    ],
)
def test_embed_sz_rule_of_thumb(rule):

    embed_cols = ["col1", "col2"]
    df = pd.DataFrame(
        {
            "col1": np.random.randint(10, size=100),
            "col2": np.random.randint(20, size=100),
        }
    )
    n_cats = {c: df[c].nunique() for c in ["col1", "col2"]}
    embed_szs = {c: embed_sz_rule(nc, embedding_rule=rule) for c, nc in n_cats.items()}
    tab_preprocessor = TabPreprocessor(cat_embed_cols=embed_cols, embedding_rule=rule)
    tdf = tab_preprocessor.fit_transform(df)  # noqa: F841
    out = [
        tab_preprocessor.embed_dim[col] == embed_szs[col] for col in embed_szs.keys()
    ]
    assert all(out)


###############################################################################
# Test Valuerror for repeated cols
###############################################################################


def test_overlapping_cols_valueerror():

    embed_cols = ["col1", "col2"]
    cont_cols = ["col1", "col2"]

    with pytest.raises(ValueError):
        tab_preprocessor = TabPreprocessor(  # noqa: F841
            cat_embed_cols=embed_cols, continuous_cols=cont_cols
        )


###############################################################################
# Test get_kernel_window
###############################################################################


def test_get_kernel_window():
    assert get_kernel_window().shape[0] == 5


###############################################################################
# Test find_bin
###############################################################################


@pytest.mark.parametrize(
    "bin_edges, values",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([-1, 0.5, 1, 2.5, 5, 6])),
        (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([-1, 0.5, 1, 2.5, 5, 6])),
    ],
)
def test_find_bin(bin_edges, values):
    if type(bin_edges) == np.ndarray and type(values) == np.ndarray:
        assert np.array_equal(find_bin(bin_edges, values), np.array([1, 1, 1, 2, 4, 4]))
    elif type(bin_edges) == torch.Tensor and type(values) == torch.Tensor:
        assert torch.equal(
            find_bin(bin_edges, values, ret_value=False),
            torch.tensor([0, 0, 0, 1, 3, 3]),
        )
