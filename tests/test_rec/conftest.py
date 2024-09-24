import pytest


@pytest.fixture(scope="module")
def cat_embed_cols():
    return ["user_id", "item_id", "user_location", "item_category"]


@pytest.fixture(scope="module")
def continuous_cols():
    return ["user_age", "item_price"]
