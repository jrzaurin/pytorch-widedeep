from torch import Tensor, nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.self_supervised._random_obfuscator import (
    RandomObfuscator,
)


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        encoder: ModelWithoutAttention,
        decoder: nn.Module,
        masked_prob: float,
    ):
        super(EncoderDecoderModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.masker = RandomObfuscator(p=masked_prob)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.encoder.is_tabnet:
            return self._forward_tabnet(X)
        else:
            return self._forward(X)

    def _forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        x_embed = self.encoder._get_embeddings(X)

        if self.training:
            masked_x, mask = self.masker(x_embed)
            x_enc = self.encoder(X)
            x_embed_rec = self.decoder(x_enc)
        else:
            x_embed_rec = self.decoder(self.encoder(X))
            mask = torch.ones(x_embed.shape).to(X.device)

        return x_embed, x_embed_rec, mask

    def _forward_tabnet(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        x_embed = self.encoder._get_embeddings(X)

        if self.training:
            masked_x, mask = self.masker(x_embed)
            prior = 1 - mask
            steps_out, _ = self.encoder(masked_x, prior=prior)
            x_embed_rec = self.decoder(steps_out)
        else:
            steps_out, _ = self.encoder(x_embed)
            x_embed_rec = self.decoder(steps_out)
            mask = torch.ones(x_embed.shape).to(X.device)

        return x_embed_rec, x_embed, mask
