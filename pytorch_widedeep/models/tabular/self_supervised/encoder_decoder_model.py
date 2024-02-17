import inspect

import torch
from torch import nn

from pytorch_widedeep.models import (
    TabMlp,
    TabNet,
    TabResnet,
    TabMlpDecoder,
    TabNetDecoder,
    TabResnetDecoder,
)
from pytorch_widedeep.wdtypes import (
    Tuple,
    Tensor,
    Optional,
    ModelWithoutAttention,
    DecoderWithoutAttention,
)
from pytorch_widedeep.models.tabular.self_supervised._random_obfuscator import (
    RandomObfuscator,
)


class EncoderDecoderModel(nn.Module):
    r"""This Class, which is referred as a 'Model', implements an Encoder-Decoder
    Self Supervised 'routine' inspired by `TabNet: Attentive Interpretable
    Tabular Learning <https://arxiv.org/abs/1908.07442>_`

    This class is in principle not exposed to the user and its documentation
    is detailed in its corresponding Trainer:  see
    ``pytorch_widedeep.self_supervised_training.EncoderDecoderTrainer``
    """

    def __init__(
        self,
        encoder: ModelWithoutAttention,
        decoder: Optional[DecoderWithoutAttention],
        masked_prob: float,
    ):
        super(EncoderDecoderModel, self).__init__()

        self.encoder = encoder

        if decoder is None:
            self.decoder = self._build_decoder(encoder)
        else:
            self.decoder = decoder
        self.masker = RandomObfuscator(p=masked_prob)

        self.is_tabnet = isinstance(self.encoder, TabNet)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.is_tabnet:
            return self._forward_tabnet(X)
        else:
            return self._forward(X)

    def _forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_embed = self.encoder._get_embeddings(X)

        if self.training:
            masked_x, mask = self.masker(x_embed)
            x_embed_rec = self.decoder(self.encoder(X))
        else:
            x_embed_rec = self.decoder(self.encoder(X))
            mask = torch.ones(x_embed.shape).to(X.device)

        return x_embed, x_embed_rec, mask

    def _forward_tabnet(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_embed = self.encoder._get_embeddings(X)

        if self.training:
            masked_x, mask = self.masker(x_embed)
            prior = 1 - mask
            steps_out, _ = self.encoder.encoder(masked_x, prior=prior)
            x_embed_rec = self.decoder(steps_out)
        else:
            steps_out, _ = self.encoder(x_embed)
            x_embed_rec = self.decoder(steps_out)
            mask = torch.ones(x_embed.shape).to(X.device)

        return x_embed_rec, x_embed, mask

    def _build_decoder(self, encoder: ModelWithoutAttention) -> DecoderWithoutAttention:
        if isinstance(encoder, TabMlp):
            decoder = self._build_tabmlp_decoder()
        if isinstance(encoder, TabResnet):
            decoder = self._build_tabresnet_decoder()
        if isinstance(encoder, TabNet):
            decoder = self._build_tabnet_decoder()
        return decoder

    def _build_tabmlp_decoder(self) -> DecoderWithoutAttention:
        common_params = (
            inspect.signature(TabMlp).parameters.keys()
            & inspect.signature(TabMlpDecoder).parameters.keys()
        )

        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)

        decoder_param["mlp_hidden_dims"] = decoder_param["mlp_hidden_dims"][::-1]
        decoder_param["embed_dim"] = (
            self.encoder.cat_out_dim + self.encoder.cont_out_dim
        )

        return TabMlpDecoder(**decoder_param)

    def _build_tabresnet_decoder(self) -> DecoderWithoutAttention:
        common_params = (
            inspect.signature(TabResnet).parameters.keys()
            & inspect.signature(TabResnetDecoder).parameters.keys()
        )

        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)

        decoder_param["blocks_dims"] = decoder_param["blocks_dims"][::-1]
        if decoder_param["mlp_hidden_dims"] is not None:
            decoder_param["mlp_hidden_dims"] = decoder_param["mlp_hidden_dims"][::-1]
        decoder_param["embed_dim"] = (
            self.encoder.cat_out_dim + self.encoder.cont_out_dim
        )

        return TabResnetDecoder(**decoder_param)

    def _build_tabnet_decoder(self) -> DecoderWithoutAttention:
        common_params = (
            inspect.signature(TabNet).parameters.keys()
            & inspect.signature(TabNetDecoder).parameters.keys()
        )

        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)

        decoder_param["embed_dim"] = (
            self.encoder.cat_out_dim + self.encoder.cont_out_dim
        )

        return TabNetDecoder(**decoder_param)
