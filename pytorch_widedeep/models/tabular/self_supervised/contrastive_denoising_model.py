import torch
from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
    ModelWithAttention,
)
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.preprocessing.tab_preprocessor import TabPreprocessor
from pytorch_widedeep.models.tabular.self_supervised._denoise_mlps import (
    CatSingleMlp,
    ContSingleMlp,
    CatMlpPerFeature,
    ContMlpPerFeature,
)
from pytorch_widedeep.models.tabular.self_supervised._augmentations import (
    mix_up,
    cut_mix,
)

DenoiseMlp = Union[CatSingleMlp, ContSingleMlp, CatMlpPerFeature, ContMlpPerFeature]


class ContrastiveDenoisingModel(nn.Module):
    r"""This Class, which is referred as a 'Model', implements a Self
    Supervised 'routine' based on the one described in "SAINT: Improved
    Neural Networks for Tabular Data via Row Attention and Contrastive
    Pre-Training", their Figure 1.

    This class is in principle not exposed to the user and its documentation
    is detailed in its corresponding Trainer:  see
    ``pytorch_widedeep.self_supervised_training.ContrastiveDenoisingTrainer``
    """

    def __init__(
        self,
        model: ModelWithAttention,
        preprocessor: TabPreprocessor,
        loss_type: Literal["contrastive", "denoising", "both"],
        projection_head1_dims: Optional[List],
        projection_head2_dims: Optional[List],
        projection_heads_activation: str,
        cat_mlp_type: Literal["single", "multiple"],
        cont_mlp_type: Literal["single", "multiple"],
        denoise_mlps_activation: str,
    ):
        super(ContrastiveDenoisingModel, self).__init__()

        self.model = model
        self.use_cat_mlp = self._use_cat_mlp(model)

        self.loss_type = loss_type
        self.projection_head1_dims = projection_head1_dims
        self.projection_head2_dims = projection_head2_dims
        self.projection_heads_activation = projection_heads_activation
        self.cat_mlp_type = cat_mlp_type
        self.cont_mlp_type = cont_mlp_type
        self.denoise_mlps_activation = denoise_mlps_activation

        self.projection_head1, self.projection_head2 = self._set_projection_heads()
        if self.loss_type in ["denoising", "both"]:
            self._t = self._tensor_to_subtract_from_cats(preprocessor)
            (
                self.denoise_cat_mlp,
                self.denoise_cont_mlp,
            ) = self._set_cat_and_cont_denoise_mlps()

    def forward(self, X: Tensor) -> Tuple[
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
    ]:
        # "uncorrupted" branch
        embed = self.model._get_embeddings(X)
        if self.model.with_cls_token:
            embed[:, 0, :] = 0.0
        encoded = self.model.encoder(embed)

        # cut_mixed and mixed_up branch
        if self.training:
            cut_mixed = cut_mix(X)
            cut_mixed_embed = self.model._get_embeddings(cut_mixed)
            if self.model.with_cls_token:
                cut_mixed_embed[:, 0, :] = 0.0
            cut_mixed_embed_mixed_up = mix_up(cut_mixed_embed)
            encoded_ = self.model.encoder(cut_mixed_embed_mixed_up)
        else:
            encoded_ = encoded.clone()

        # projections for constrastive loss
        if self.loss_type in ["contrastive", "both"]:
            if self.model.with_cls_token:
                g_projs = (
                    self.projection_head1(encoded[:, 1:, :]),
                    self.projection_head2(encoded_[:, 1:, :]),
                )
            else:
                g_projs = (
                    self.projection_head1(encoded),
                    self.projection_head2(encoded_),
                )
        else:
            g_projs = None

        # mlps for denoising loss
        if self.loss_type in ["denoising", "both"]:
            if self._t is not None:
                self._t = self._t.to(X.device)
                _X = X - self._t.repeat(X.size(0), 1)
            else:
                _X = X

            if self.model.with_cls_token:
                _X[:, 0] = 0.0

            cat_x_and_x_ = (
                self.denoise_cat_mlp(_X, encoded_)
                if self.denoise_cat_mlp is not None
                else None
            )

            cont_x_and_x_ = (
                self.denoise_cont_mlp(_X, encoded_)
                if self.denoise_cont_mlp is not None
                else None
            )

        else:
            cat_x_and_x_, cont_x_and_x_ = None, None

        return g_projs, cat_x_and_x_, cont_x_and_x_

    def _set_projection_heads(
        self,
    ) -> Tuple[Union[MLP, nn.Identity], Union[MLP, nn.Identity]]:
        if self.projection_head1_dims is not None:
            projection_head1: Union[MLP, nn.Identity] = MLP(
                d_hidden=self.projection_head1_dims,
                activation=self.projection_heads_activation,
                dropout=0.0,
                batchnorm=False,
                batchnorm_last=False,
                linear_first=False,
            )
            if self.projection_head2_dims is not None:
                projection_head2: Union[MLP, nn.Identity] = MLP(
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

    def _set_cat_and_cont_denoise_mlps(
        self,
    ) -> Tuple[Optional[DenoiseMlp], Optional[DenoiseMlp]]:
        if self.use_cat_mlp:
            if self.cat_mlp_type == "single":
                denoise_cat_mlp: Union[CatSingleMlp, CatMlpPerFeature] = CatSingleMlp(
                    self.model.input_dim,
                    self.model.cat_embed_input,
                    self.model.column_idx,
                    self.denoise_mlps_activation,
                )
            elif self.cat_mlp_type == "multiple":
                denoise_cat_mlp = CatMlpPerFeature(
                    self.model.input_dim,
                    self.model.cat_embed_input,
                    self.model.column_idx,
                    self.denoise_mlps_activation,
                )
        else:
            denoise_cat_mlp = None

        if self.model.continuous_cols is not None:
            if self.cont_mlp_type == "single":
                denoise_cont_mlp: Union[ContSingleMlp, ContMlpPerFeature] = (
                    ContSingleMlp(
                        self.model.input_dim,
                        self.model.continuous_cols,
                        self.model.column_idx,
                        self.denoise_mlps_activation,
                    )
                )
            elif self.cont_mlp_type == "multiple":
                denoise_cont_mlp = ContMlpPerFeature(
                    self.model.input_dim,
                    self.model.continuous_cols,
                    self.model.column_idx,
                    self.denoise_mlps_activation,
                )
        else:
            denoise_cont_mlp = None

        return denoise_cat_mlp, denoise_cont_mlp

    def _set_idx_to_substract(self, encoding_dict) -> Optional[Dict[str, int]]:
        if self.cat_mlp_type == "multiple":
            idx_to_substract: Optional[Dict[str, int]] = {
                k: min(sorted(list(v.values()))) for k, v in encoding_dict.items()
            }

        if self.cat_mlp_type == "single":
            idx_to_substract = None

        return idx_to_substract

    def _tensor_to_subtract_from_cats(self, preprocessor) -> Optional[Tensor]:
        if hasattr(preprocessor, "label_encoder"):
            encoding_dict = preprocessor.label_encoder.encoding_dict
            self.idx_to_substract = self._set_idx_to_substract(encoding_dict)

            _t = torch.zeros(len(self.model.column_idx))

            if self.idx_to_substract is not None:
                for colname, idx in self.idx_to_substract.items():
                    _t[self.model.column_idx[colname]] = idx
            else:
                for colname, _ in self.model.cat_embed_input:
                    # 0 is reserved for padding, 1 for the '[CLS]' token, if present
                    _t[self.model.column_idx[colname]] = (
                        2 if self.model.with_cls_token else 1
                    )
        else:
            _t = None

        return _t

    def _use_cat_mlp(self, model):
        if model.cat_embed_input is not None:
            if (
                len(model.cat_embed_input) == 1
                and model.cat_embed_input[0][0] == "cls_token"
            ):
                return False
            else:
                return True
        else:
            return False
