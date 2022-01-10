import numpy as np
from scipy.sparse import csc_matrix

from pytorch_widedeep.wdtypes import *  # noqa: F403


def create_explain_matrix(model: WideDeep) -> csc_matrix:
    """
    Returns a sparse matrix used to compute the feature importances after
    training

    Parameters
    ----------
    model: WideDeep
        object of type ``WideDeep``

    Examples
    --------
    >>> from pytorch_widedeep.models import TabNet, WideDeep
    >>> from pytorch_widedeep.models.tabular.tabnet._utils import create_explain_matrix
    >>> embed_input = [("a", 4, 2), ("b", 4, 2), ("c", 4, 2)]
    >>> cont_cols = ["d", "e"]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c", "d", "e"])}
    >>> deeptabular = TabNet(column_idx=column_idx, cat_embed_input=embed_input, continuous_cols=cont_cols)
    >>> model = WideDeep(deeptabular=deeptabular)
    >>> reduce_mtx = create_explain_matrix(model)
    >>> reduce_mtx.todense()
    matrix([[1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]])
    """

    tabnet_backbone = list(model.deeptabular.children())[0]

    embed_out_dim: int = tabnet_backbone.embed_out_dim  # type: ignore[assignment]
    column_idx: Dict = tabnet_backbone.column_idx  # type: ignore[assignment]

    cat_setup = extract_cat_setup(tabnet_backbone)
    cont_setup = extract_cont_setup(tabnet_backbone)

    n_feat = len(cat_setup + cont_setup)

    col_embeds = {}
    embeds_colname = []

    for cats in cat_setup:
        col_embeds[cats[0]] = cats[2] - 1
        embeds_colname.append(cats[0])

    if len(cont_setup) > 0:
        if isinstance(cont_setup[0], tuple):
            for conts in cont_setup:
                col_embeds[conts[0]] = conts[1] - 1
                embeds_colname.append(conts[0])
            cont_colname = []
        else:
            cont_colname = cont_setup
    else:
        cont_colname = []

    embed_cum_counter = 0
    indices_trick = []

    for colname, idx in column_idx.items():
        if colname in cont_colname:
            indices_trick.append([idx + embed_cum_counter])
        elif colname in embeds_colname:
            indices_trick.append(
                range(  # type: ignore[arg-type]
                    idx + embed_cum_counter,
                    idx + embed_cum_counter + col_embeds[colname] + 1,
                )
            )
            embed_cum_counter += col_embeds[colname]

    reducing_matrix = np.zeros((embed_out_dim, n_feat))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return csc_matrix(reducing_matrix)


def extract_cat_setup(backbone: Module) -> List:
    cat_cols: List = backbone.cat_embed_input  # type: ignore[assignment]
    if cat_cols is not None:
        return cat_cols
    else:
        return []


def extract_cont_setup(backbone: Module) -> List:

    cont_cols: List = backbone.continuous_cols  # type: ignore[assignment]
    embed_continuous = backbone.embed_continuous

    if cont_cols is not None:
        if embed_continuous:
            cont_embed_dim = [backbone.cont_embed_dim] * len(cont_cols)
            cont_setup: List = [
                (colname, embed_dim)
                for colname, embed_dim in zip(cont_cols, cont_embed_dim)
            ]
        else:
            cont_setup = cont_cols
    else:
        cont_setup = []

    return cont_setup
