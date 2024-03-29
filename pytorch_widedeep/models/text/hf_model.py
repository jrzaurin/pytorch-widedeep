import warnings

import torch
from torch import nn
from transformers import (
    BertModel,
    BertConfig,
    AlbertModel,
    AlbertConfig,
    ElectraModel,
    RobertaModel,
    ElectraConfig,
    RobertaConfig,
    DistilBertModel,
    PreTrainedModel,
    DistilBertConfig,
    PretrainedConfig,
)

from pytorch_widedeep.wdtypes import Tuple, Tensor, Optional
from pytorch_widedeep.models.text.hf_utils import get_model_class


class HFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        use_cls_token: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_class = get_model_class(model_name)
        if self.model_class == "electra" and use_cls_token is not None:
            if use_cls_token:
                raise ValueError(
                    "Electra does not use a cls token to aggregate representations"
                )
            else:
                warnings.warn(
                    "Electra does not use a cls token to aggregate representations. "
                    "The use_cls_token argument will be ignored."
                )

        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.kwargs = kwargs

        self.config, self.model = self._get_model_objects(
            self.model_name, self.model_class
        )

        self.output_attention_weights = kwargs.get("output_attentions", False)

    def _get_model_objects(
        self, model_name: str, model_class: str
    ) -> Tuple[PretrainedConfig, PreTrainedModel]:
        if model_class == "distilbert":
            config = DistilBertConfig.from_pretrained(model_name)
            model = DistilBertModel(config)
        elif model_class == "bert":
            config = BertConfig.from_pretrained(model_name)
            model = BertModel(config)
        elif model_class == "roberta":
            config = RobertaConfig.from_pretrained(model_name)
            model = RobertaModel(config)
        elif model_class == "albert":
            config = AlbertConfig.from_pretrained(model_name)
            model = AlbertModel(config)
        elif model_class == "electra":
            config = ElectraConfig.from_pretrained(model_name)
            model = ElectraModel(config)
        else:
            raise ValueError(
                f"model_class should be one of 'distilbert', 'bert', 'roberta', 'albert', or 'electra'. Got {model_class}"
            )
        return config, model

    def forward(self, X: Tensor) -> Tensor:

        # this is inefficient since the attention mask is returned by the
        # tokenizer, but all models in this library use a forward pass that
        # takes ONLY an input tensor. A fix will be addressed in a future
        attn_mask = (X != 0).type(torch.int8)

        output = self.model(input_ids=X, attention_mask=attn_mask, **self.kwargs)

        if self.output_attention_weights:
            self.attn_weights = output["attentions"]

        if self.use_cls_token:
            output = output[0][:, 0, :]

        return output[0]

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size

    @property
    def attention_weight(self) -> Tensor:
        if not self.output_attention_weights:
            raise AttributeError(
                "The output_attention_weights attribute was not set to True when creating the model object "
                "Please pass an output_attention_weights=True argument when creating the HFModel object"
            )
        return self.attn_weights


if __name__ == "__main__":

    from pytorch_widedeep.models.text.hf_tokenizer import HFTokenizer

    texts = ["this is a text", "Using just a few words to create a sentence"]

    tokenizer = HFTokenizer(
        "distilbert-base-uncased",
        use_fast_tokenizer=False,
        num_workers=1,
    )

    X_text_arr = tokenizer.encode(
        texts,
        max_length=10,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )

    model = HFModel(model_name="distilbert-base-uncased", output_attentions=True)

    X_text = torch.tensor(X_text_arr)
    out = model(X_text)
