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

        self.projection_head1, self.projection_head2 = self._set_projection_heads()
        (
            self.denoise_cat_mlp,
            self.denoise_cont_mlp,
        ) = self._set_cat_and_cont_denoise_mlps()

    def forward(
        self, X: Tensor
    ) -> Tuple[
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
    ]:

        encoded = self.model.encoder(self.model._get_embeddings(X))

        cut_mixed = cut_mix(X)
        cut_mixed_embed = self.model._get_embeddings(cut_mixed)
        cut_mixed_embed_mixed_up = mix_up(cut_mixed_embed)
        encoded_ = self.model.encoder(cut_mixed_embed_mixed_up)

        if self.loss_type in ["contrastive", "both"]:
            g_projs = (self.projection_head1(encoded), self.projection_head2(encoded_))
        else:
            g_projs = None

        if self.loss_type in ["denoising", "both"]:
            if self.model.cat_embed_input is not None:
                cat_x_and_x_ = self.denoise_cat_mlp(X, encoded_)
            else:
                cat_x_and_x_ = None
            if self.model.continuous_cols is not None:
                cont_x_and_x_ = self.denoise_cont_mlp(X, encoded_)
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
            denoise_cat_mlp = CatSingleMlp(self.model, self.denoise_mlps_activation)
        elif self.cat_mlp_type == "multiple":
            denoise_cat_mlp = CatFeaturesMlp(self.model, self.denoise_mlps_activation)

        if self.cont_mlp_type == "single":
            denoise_cont_mlp = ContSingleMlp(self.model, self.denoise_mlps_activation)
        elif self.cont_mlp_type == "multiple":
            denoise_cont_mlp = ContFeaturesMlp(self.model, self.denoise_mlps_activation)

        return denoise_cat_mlp, denoise_cont_mlp
