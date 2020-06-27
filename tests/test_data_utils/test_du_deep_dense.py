import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.preprocessing import DeepPreprocessor
from pytorch_widedeep.utils.dense_utils import LabelEncoder, label_encoder


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
# Simple test of functionality: testing the label_encoder function
###############################################################################
le_letters = LabelEncoder(["col1", "col2"])
df_letters_le = le_letters.fit_transform(df_letters)
le_numbers = LabelEncoder(["col1", "col2"])
df_numbers_le = le_numbers.fit_transform(df_numbers)


@pytest.mark.parametrize(
    "input_df, encoding_dict, output_df",
    [(df_letters, le_letters, df_letters_le), (df_numbers, le_numbers, df_numbers_le),],
)
def test_label_encoder(input_df, encoder, output_df):
    original_df = encoder.inverse_transform(output_df)
    assert original_df.equals(input_df)


###############################################################################
# Simple test of functionality: testing the label_encoder function
###############################################################################
df_letters_le, letters_enconding_dict = label_encoder(df_letters, ["col1", "col2"])
df_numbers_le, numbers_enconding_dict = label_encoder(df_numbers, ["col1", "col2"])


@pytest.mark.parametrize(
    "input_df, encoding_dict, output_df",
    [
        (df_letters, letters_enconding_dict, df_letters_le),
        (df_numbers, numbers_enconding_dict, df_numbers_le),
    ],
)
def test_label_encoder(input_df, encoding_dict, output_df):
    tmp_df = input_df.copy()
    for c in input_df.columns:
        tmp_df[c] = tmp_df[c].map(encoding_dict[c])
    assert tmp_df.equals(output_df)


################################################################################
# Same as before but testing functioning when passed a custom encoding dict
###############################################################################
encoding_dict_1 = {
    c: {k: v for v, k in enumerate(sorted(df_letters[c].unique()))}
    for c in df_letters.columns
}
encoding_dict_2 = {
    c: {k: v for v, k in enumerate(sorted(df_numbers[c].unique()))}
    for c in df_numbers.columns
}

df_letters_le, letters_enconding_dict = label_encoder(
    df_letters, cols=["col1", "col2"], val_to_idx=encoding_dict_1
)
df_numbers_le, numbers_enconding_dict = label_encoder(
    df_numbers, cols=["col1", "col2"], val_to_idx=encoding_dict_2
)


@pytest.mark.parametrize(
    "input_df, encoding_dict, output_df",
    [
        (df_letters, encoding_dict_1, df_letters_le),
        (df_numbers, encoding_dict_2, df_numbers_le),
    ],
)
def test_label_encoder_with_custom_encoder(input_df, encoding_dict, output_df):
    tmp_df = input_df.copy()
    for c in input_df.columns:
        tmp_df[c] = tmp_df[c].map(encoding_dict[c])
    assert tmp_df.equals(output_df)


################################################################################
# Test the DeepPreprocessor: only categorical columns to be represented with
# embeddings
###############################################################################
cat_embed_cols = [("col1", 5), ("col2", 5)]

preprocessor1 = DeepPreprocessor(cat_embed_cols)
X_letters = preprocessor1.fit_transform(df_letters)
embed_input_letters = preprocessor1.embeddings_input
decoding_dict_letters = {
    c: {k: v for v, k in preprocessor1.encoding_dict[c].items()}
    for c in preprocessor1.encoding_dict.keys()
}

preprocessor2 = DeepPreprocessor(cat_embed_cols)
X_numbers = preprocessor2.fit_transform(df_numbers)
embed_input_numbers = preprocessor2.embeddings_input
decoding_dict_numbers = {
    c: {k: v for v, k in preprocessor2.encoding_dict[c].items()}
    for c in preprocessor2.encoding_dict.keys()
}


errors = []


@pytest.mark.parametrize(
    "input_df, X_deep, embed_input, decoding_dict, error_list",
    [
        (df_letters, X_letters, embed_input_letters, decoding_dict_letters, errors),
        (df_numbers, X_numbers, embed_input_numbers, decoding_dict_numbers, errors),
    ],
)
def test_prepare_deep_without_continous_columns(
    input_df, X_deep, embed_input, decoding_dict, error_list
):
    for i, c in enumerate(input_df.columns):
        if (
            input_df[c].nunique() != embed_input[i][1]
            or cat_embed_cols[i][1] != embed_input[i][2]
        ):
            error_list.append(
                "error: the setup output does not match the intended input"
            )

    tmp_df = pd.DataFrame({"col1": X_deep[:, 0], "col2": X_deep[:, 1]})
    for c in input_df.columns:
        tmp_df[c] = tmp_df[c].map(decoding_dict[c])

    if not tmp_df.equals(input_df):
        error_list.append("error: the decoding does not match the encoding")

    assert not error_list, "errors occured:\n{}".format("\n".join(error_list))


################################################################################
# Test the DeepPreprocessor: only continouos columns
###############################################################################
def test_prepare_deep_without_embedding_columns():

    errors = []
    df_randint = pd.DataFrame(np.random.choice(np.arange(100), (100, 2)))
    df_randint.columns = ["col1", "col2"]
    preprocessor3 = DeepPreprocessor(continuous_cols=["col1", "col2"])

    try:
        X_randint = preprocessor3.fit_transform(df_randint)
    except:
        errors.append("Fundamental Error")

    out_booleans = []

    means, stds = np.mean(X_randint, axis=0), np.std(X_randint, axis=0)
    for mean, std in zip(means, stds):
        out_booleans.append(np.isclose(mean, 0.0))
        out_booleans.append(np.isclose(std, 1.0))

    if not np.all(out_booleans):
        errors.append("There is something going on with the scaler")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))
