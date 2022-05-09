from torch import Tensor, nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.self_supervised_training._denoise_mlps import (
    CatSingleMlp,
    ContSingleMlp,
    CatFeaturesMlp,
    ContFeaturesMlp,
)
from pytorch_widedeep.self_supervised_training._augmentations import (
    mix_up,
    cut_mix,
)


class SelfSupervisedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        encoding_dict: Dict[str, Dict[str, int]],
        loss_type: Literal["contrastive", "denoising", "both"],
        projection_head1_dims: Optional[List],
        projection_head2_dims: Optional[List],
        projection_heads_activation: str,
        cat_mlp_type: Literal["single", "multiple"],
        cont_mlp_type: Literal["single", "multiple"],
        denoise_mlps_activation: str,
    ):
        super(SelfSupervisedModel, self).__init__()

        self.model = model
        self.loss_type = loss_type
        self.projection_head1_dims = projection_head1_dims
        self.projection_head2_dims = projection_head2_dims
        self.projection_heads_activation = projection_heads_activation
        self.cat_mlp_type = cat_mlp_type
        self.cont_mlp_type = cont_mlp_type
        self.denoise_mlps_activation = denoise_mlps_activation

        self.cat_embed_input, self.column_idx = self._adjust_if_with_cls_token(
            encoding_dict
        )

        self.projection_head1, self.projection_head2 = self._set_projection_heads()
        (
            self.denoise_cat_mlp,
            self.denoise_cont_mlp,
        ) = self._set_cat_and_cont_denoise_mlps()

        self._t = self._tensor_to_subtract(encoding_dict)

    def forward(
        self, X: Tensor
    ) -> Tuple[
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
    ]:

        _X = self._prepare_x(X)

        embed = self.model._get_embeddings(X)
        _embed = embed[:, 1:] if self.model.with_cls_token else embed

        encoded = self.model.encoder(_embed)

        cut_mixed = cut_mix(X)
        cut_mixed_embed = self.model._get_embeddings(cut_mixed)
        _cut_mixed_embed = (
            cut_mixed_embed[:, 1:] if self.model.with_cls_token else cut_mixed_embed
        )
        cut_mixed_embed_mixed_up = mix_up(_cut_mixed_embed)
        encoded_ = self.model.encoder(cut_mixed_embed_mixed_up)

        if self.loss_type in ["contrastive", "both"]:
            g_projs = (self.projection_head1(encoded), self.projection_head2(encoded_))
        else:
            g_projs = None

        if self.loss_type in ["denoising", "both"]:
            if self.model.cat_embed_input is not None:
                cat_x_and_x_ = self.denoise_cat_mlp(_X, encoded_)
            else:
                cat_x_and_x_ = None
            if self.model.continuous_cols is not None:
                cont_x_and_x_ = self.denoise_cont_mlp(_X, encoded_)
            else:
                cont_x_and_x_ = None

        return g_projs, cat_x_and_x_, cont_x_and_x_

    def _set_projection_heads(self) -> Tuple[nn.Module, nn.Module]:

        if self.projection_head1_dims is not None:
            projection_head1 = MLP(
                d_hidden=self.projection_head1_dims,
                activation=self.projection_heads_activation,
                dropout=0.0,
                batchnorm=False,
                batchnorm_last=False,
                linear_first=False,
            )
            if self.projection_head2_dims is not None:
                projection_head2 = MLP(
                    d_hidden=self.projection_head2_dims,
                    activation=self.projection_heads_activation,
                    dropout=0.0,
                    batchnorm=False,
                    batchnorm_last=False,
                    linear_first=False,
                )
            else:
                projection_head2 = projection_head1
        else:
            projection_head1 = nn.Identity()
            projection_head2 = nn.Identity()

        return projection_head1, projection_head2

    def _set_cat_and_cont_denoise_mlps(self) -> Tuple[nn.Module, nn.Module]:

        if self.cat_mlp_type == "single":
            denoise_cat_mlp = CatSingleMlp(
                self.model.input_dim,
                self.cat_embed_input,
                self.column_idx,
                self.denoise_mlps_activation,
            )
        elif self.cat_mlp_type == "multiple":
            denoise_cat_mlp = CatFeaturesMlp(
                self.model.input_dim,
                self.cat_embed_input,
                self.column_idx,
                self.denoise_mlps_activation,
            )

        if self.cont_mlp_type == "single":
            denoise_cont_mlp = ContSingleMlp(
                self.model.input_dim,
                self.model.continuous_cols,
                self.column_idx,
                self.denoise_mlps_activation,
            )
        elif self.cont_mlp_type == "multiple":
            denoise_cont_mlp = ContFeaturesMlp(
                self.model.input_dim,
                self.model.continuous_cols,
                self.column_idx,
                self.denoise_mlps_activation,
            )

        return denoise_cat_mlp, denoise_cont_mlp

    def _prepare_x(self, X_tab: Tensor) -> Tensor:

        _X_tab = X_tab[:, 1:] if self.model.with_cls_token else X_tab

        return _X_tab - self._t.repeat(X_tab.size(0), 1)

    def _adjust_if_with_cls_token(self, encoding_dict):
        if self.model.with_cls_token:
            adj_column_idx = {
                k: self.model.column_idx[k] - 1
                for k in self.model.column_idx
                if k != "cls_token"
            }
            adj_cat_embed_input = self.model.cat_embed_input[1:]
        else:
            adj_column_idx = dict(column_idx)
            adj_cat_embed_input = self.model.cat_embed_input

        return adj_cat_embed_input, adj_column_idx

    def _set_idx_to_substract(self, encoding_dict) -> Tensor:

        if self.model.with_cls_token:
            adj_encoding_dict = {
                k: v for k, v in encoding_dict.items() if k != "cls_token"
            }

        if self.cat_mlp_type == "multiple":
            idx_to_substract: Optional[Dict[str, Dict[str, int]]] = {
                k: min(sorted(list(v.values()))) for k, v in adj_encoding_dict.items()
            }

        if self.cat_mlp_type == "single":
            idx_to_substract = None

        return idx_to_substract

    def _tensor_to_subtract(self, encoding_dict) -> Tensor:

        self.idx_to_substract = self._set_idx_to_substract(encoding_dict)

        _t = torch.zeros(len(self.column_idx))

        if self.idx_to_substract is not None:
            for colname, idx in self.idx_to_substract.items():
                _t[self.column_idx[colname]] = idx
        else:
            for colname, _ in self.cat_embed_input:
                # 0 is reserved for padding, 1 for the '[CLS]' token, if present
                _t[self.column_idx[colname]] = 2 if self.model.with_cls_token else 1
        return _t
