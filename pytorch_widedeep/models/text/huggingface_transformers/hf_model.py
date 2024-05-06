import warnings

import torch

from pytorch_widedeep.wdtypes import List, Tensor, Optional
from pytorch_widedeep.utils.hf_utils import (
    get_model_class,
    get_config_and_model,
)
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models._base_wd_model_component import (
    BaseWDModelComponent,
)


class HFModel(BaseWDModelComponent):
    """This class is a wrapper around the Hugging Face transformers library. It
    can be used as the text component of a Wide & Deep model or independently
    by itself.

    At the moment only models from the families BERT, RoBERTa, DistilBERT,
    ALBERT and ELECTRA are supported. This is because this library is
    designed to address classification and regression tasks and these are the
    most 'popular' encoder-only models, which have proved to be those that
    work best for these tasks.

    Parameters
    ----------
    model_name: str
        The model name from the transformers library e.g. 'bert-base-uncased'.
        Currently supported models are those from the families: BERT, RoBERTa,
        DistilBERT, ALBERT and ELECTRA.
    use_cls_token: bool, default = True
        Boolean indicating whether to use the [CLS] token or the mean of the
        sequence of hidden states as the sentence embedding
    trainable_parameters: List, Optional, default = None
        List with the names of the model parameters that will be trained. If
        None, none of the parameters will be trainable
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: _[128, 64]_
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in the
        dense layers that form the head
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`
    verbose: bool, default = False
        If True, it will print information about the model
    **kwargs
        Additional kwargs to be passed to the model

    Attributes
    ----------
    head: nn.Module
        Stack of dense layers on top of the transformer. This will only exists
        if `head_layers_dim` is not None

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import HFModel
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1).long()
    >>> model = HFModel(model_name='bert-base-uncased')
    >>> out = model(X_text)
    """

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

        # TO DO: add warning regarging ELECTRA as ELECTRA does not have a cls
        # token.  Research what happens with ELECTRA
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
                "was run with add_special_tokens=True",
                UserWarning,
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
            # Here one can choose to flatten, but unless the sequence length
            # is very small, flatten will result in a very large output
            # tensor.
            output = output[0].mean(dim=1)

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
        r"""Returns the attention weights if the model was created with the
        output_attention_weights=True argument. If not, it will raise an
        AttributeError.

        The shape of the attention weights is $(N, H, F, F)$, where $N$ is the
        batch size, $H$ is the number of attention heads and $F$ is the
        sequence length.
        """
        if not self.output_attention_weights:
            raise AttributeError(
                "The output_attention_weights attribute was not set to True when creating the model object "
                "Please pass an output_attention_weights=True argument when creating the HFModel object"
            )
        return self.attn_weights
