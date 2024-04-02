import warnings

import torch
from torch import nn

from pytorch_widedeep.wdtypes import List, Tensor, Optional
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.text.huggingface_transformers.hf_utils import (
    get_model_class,
    get_config_and_model,
)


class HFModel(nn.Module):
    @alias("use_cls_token", ["use_special_token"])
    def __init__(
        self,
        model_name: str,
        use_cls_token: bool = True,
        trainable_parameters: Optional[List[str]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: Optional[float] = None,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        # TO DO: add warning regarging electra as electra does not have a cls
        # token.  Research what happens with Electra
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.trainable_parameters = trainable_parameters
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first
        self.verbose = verbose
        self.kwargs = kwargs

        if self.verbose and self.use_cls_token:
            warnings.warn(
                "The model will use the [CLS] token. Make sure the tokenizer "
                "was run with add_special_tokens=True"
            )

        self.model_class = get_model_class(model_name)

        self.config, self.model = get_config_and_model(self.model_name)

        self.output_attention_weights = kwargs.get("output_attentions", False)

        if self.trainable_parameters is not None:
            for n, p in self.model.named_parameters():
                p.requires_grad = any([tl in n for tl in self.trainable_parameters])

        # FC-Head (Mlp). Note that the FC head will always be trainable
        if self.head_hidden_dims is not None:
            head_hidden_dims = [self.config.hidden_size] + self.head_hidden_dims
            self.head = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )

    def forward(self, X: Tensor) -> Tensor:

        # this is inefficient since the attention mask is returned by the
        # tokenizer, but all models in this library use a forward pass that
        # takes ONLY an input tensor. A fix will be addressed in a future
        attn_mask = (X != 0).type(torch.int8)

        output = self.model(input_ids=X, attention_mask=attn_mask, **self.kwargs)

        if self.output_attention_weights:
            # TO CONSIDER: attention weights as a returned object and not an
            # attribute
            self.attn_weights = output["attentions"]

        if self.use_cls_token:
            output = output[0][:, 0, :]
        else:
            output = output[0]

        if self.head_hidden_dims is not None:
            output = self.head(output)

        return output

    @property
    def output_dim(self) -> int:
        return (
            self.head_hidden_dims[-1]
            if self.head_hidden_dims is not None
            else self.config.hidden_size
        )

    @property
    def attention_weight(self) -> Tensor:
        if not self.output_attention_weights:
            raise AttributeError(
                "The output_attention_weights attribute was not set to True when creating the model object "
                "Please pass an output_attention_weights=True argument when creating the HFModel object"
            )
        return self.attn_weights


if __name__ == "__main__":

    from pytorch_widedeep.models.text.huggingface_transformers.hf_tokenizer import (
        HFTokenizer,
    )

    texts = ["this is a text", "Using just a few words to create a sentence"]

    tokenizer = HFTokenizer(
        model_name="google/electra-base-discriminator",
        # model_name="distilbert-base-uncased",
        use_fast_tokenizer=False,
        num_workers=1,
    )

    X_text_arr = tokenizer.encode(
        texts,
        max_length=15,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    model = HFModel(
        model_name="google/electra-base-discriminator",
        use_cls_token=True,
        # model_name="distilbert-base-uncased",
        # trainable_parameters=["layer.5"],
        output_attentions=True,
    )

    X_text = torch.tensor(X_text_arr)
    out = model(X_text)
