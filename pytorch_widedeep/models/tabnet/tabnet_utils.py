import numpy as np
import scipy

from pytorch_widedeep.wdtypes import WideDeep


def create_explain_matrix(model: WideDeep):

    (
        embed_input,
        column_idx,
        embed_and_cont_dim,
    ) = _extract_tabnet_params(model)

    n_feat = len(column_idx)
    col_embeds = {e[0]: e[2] - 1 for e in embed_input}
    embed_colname = [e[0] for e in embed_input]
    cont_colname = [c for c in column_idx.keys() if c not in embed_colname]

    embed_cum_counter = 0
    indices_trick = []
    for colname, idx in column_idx.items():
        if colname in cont_colname:
            indices_trick.append([idx + embed_cum_counter])
        elif colname in embed_colname:
            indices_trick.append(
                range(  # type: ignore[arg-type]
                    idx + embed_cum_counter,
                    idx + embed_cum_counter + col_embeds[colname] + 1,
                )
            )
            embed_cum_counter += col_embeds[colname]

    reducing_matrix = np.zeros((embed_and_cont_dim, n_feat))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)


def _extract_tabnet_params(model: WideDeep):

    tabnet_backbone = list(model.deeptabular.children())[0]

    column_idx = tabnet_backbone.column_idx
    embed_input = tabnet_backbone.embed_input
    embed_and_cont_dim = tabnet_backbone.embed_and_cont_dim

    return embed_input, column_idx, embed_and_cont_dim
