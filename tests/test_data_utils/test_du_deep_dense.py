import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.preprocessing import DensePreprocessor
from pytorch_widedeep.utils.dense_utils import LabelEncoder


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
# Test the DensePreprocessor: only categorical columns to be represented with
# embeddings
###############################################################################
cat_embed_cols = [("col1", 5), ("col2", 5)]

preprocessor1 = DensePreprocessor(cat_embed_cols)  # type: ignore[arg-type]
X_letters = preprocessor1.fit_transform(df_letters)

preprocessor2 = DensePreprocessor(cat_embed_cols)  # type: ignore[arg-type]
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
            input_df[c].nunique() + 1 != preprocessor.embeddings_input[i][1]
            or cat_embed_cols[i][1] != preprocessor.embeddings_input[i][2]
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
# Test the DensePreprocessor: only continouos columns
###############################################################################
def test_prepare_deep_without_embedding_columns():

    errors = []
    df_randint = pd.DataFrame(np.random.choice(np.arange(100), (100, 2)))
    df_randint.columns = ["col1", "col2"]
    preprocessor3 = DensePreprocessor(continuous_cols=["col1", "col2"])

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
