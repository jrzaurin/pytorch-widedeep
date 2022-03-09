import warnings

import torchvision
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.image._layers import conv_layer
from pytorch_widedeep.models.tabular.mlp._layers import MLP

allowed_pretrained_models = [
    "resnet",
    "shufflenet",
    "resnext",
    "wide_resnet",
    "regnet",
    "densenet",
    "mobilenet",
    "mnasnet",
    "efficientnet",
    "squeezenet",
]


class Vision(nn.Module):
    r"""Defines a standard image classifier/regressor using a pretrained
    network or a sequence of convolution layers that can be used as the
    ``deepimage`` component of a Wide & Deep model or independently by
    itself.

    Parameters
    ----------
    pretrained_model_name: Optional, str, default = None
        Name of the pretrained model. Should be a variant of the following
        architectures: `'resnet`', `'shufflenet`', `'resnext`',
        `'wide_resnet`', `'regnet`', `'densenet`', `'mobilenetv3`',
        `'mobilenetv2`', `'mnasnet`', `'efficientnet`' and `'squeezenet`'. if
        `pretrained_model_name = None` a basic, fully trainable CNN will be
        used.
    n_trainable: Optional, int, default = None
        Number of trainable layers starting from the layer closer to the
        output neuron(s). Note that this number DOES NOT take into account
        the so-called "head" which is ALWAYS trainable. If
        ``trainable_params`` is not None this parameter will be ignored
    trainable_params: Optional, list, default = None
        List of strings containing the names (or substring within the name) of
        the parameters that will be trained. For example, if we use a
        `'resnet18'` pretrainable model and we set ``trainable_params =
        ['layer4']`` only the parameters of `'layer4'` of the network (and the
        head, as mentioned before) will be trained. Note that setting this or
        the previous parameter involves some knowledge of the architecture
        used.
    channel_sizes: list, default = [64, 128, 256, 512]
        List of integers with the channel sizes of a CNN in case we choose not
        to use a pretrained model
    kernel_sizes: list or int, default = 3
        List of integers with the kernel sizes of a CNN in case we choose not
        to use a pretrained model. Must be of length equal to `len
        (channel_sizes) - 1`.
    strides: list or int, default = 1
        List of integers with the stride sizes of a CNN in case we choose not
        to use a pretrained model. Must be of length equal to `len
        (channel_sizes) - 1`.
    head_hidden_dims: Optional, list, default = None
        List with the number of neurons per dense layer in the head. e.g: [64,32]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    head_dropout: float, default = 0.1
        float indicating the dropout between the dense layers.
    head_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    head_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    features: ``nn.Module``
        The pretrained model or Standard CNN plus the optional head
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the ``WideDeep`` class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Vision
    >>> X_img = torch.rand((2,3,224,224))
    >>> model = Vision(channel_sizes=[64, 128], kernel_sizes = [3, 3], strides=[1, 1], head_hidden_dims=[32, 8])
    >>> out = model(X_img)
    """

    def __init__(
        self,
        pretrained_model_name: Optional[str] = None,
        n_trainable: Optional[int] = None,
        trainable_params: Optional[List[str]] = None,
        channel_sizes: List[int] = [64, 128, 256, 512],
        kernel_sizes: Union[int, List[int]] = [7, 3, 3, 3],
        strides: Union[int, List[int]] = [2, 1, 1, 1],
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: Union[float, List[float]] = 0.1,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
    ):
        super(Vision, self).__init__()

        self.pretrained_model_name = pretrained_model_name
        self.n_trainable = n_trainable
        self.trainable_params = trainable_params
        self.channel_sizes = channel_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first

        if pretrained_model_name is not None:
            valid_pretrained_model_name = any(
                [name in pretrained_model_name for name in allowed_pretrained_models]
            )
            if not valid_pretrained_model_name:
                raise ValueError(
                    f"{pretrained_model_name} is not among the allowed pretrained models."
                    f" These are {allowed_pretrained_models}. Please choose a variant of these architectures"
                )
            if n_trainable is not None and trainable_params is not None:
                raise UserWarning(
                    "Both 'n_trainable' and 'trainable_params' are not None. 'trainable_params' will be used"
                )

        self.features, output_dim = self._get_features()

        if pretrained_model_name is not None:
            self._freeze(self.features)

        if self.head_hidden_dims is not None:
            head_hidden_dims = [output_dim] + self.head_hidden_dims
            self.vision_mlp = MLP(
                head_hidden_dims,
                self.head_activation,
                self.head_dropout,
                self.head_batchnorm,
                self.head_batchnorm_last,
                self.head_linear_first,
            )
            output_dim = self.head_hidden_dims[-1]  # type: ignore[assignment]

        self.output_dim = output_dim

    def forward(self, X: Tensor) -> Tensor:

        x = self.features(X)

        if len(x.shape) > 2:
            if x.shape[2] > 1:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        if self.head_hidden_dims is not None:
            x = self.vision_mlp(x)

        return x

    def _get_features(self) -> Tuple[nn.Module, int]:
        if self.pretrained_model_name is not None:
            pretrained_model = torchvision.models.__dict__[self.pretrained_model_name](
                pretrained=True
            )
            output_dim: int = self.get_backbone_output_dim(pretrained_model)
            features = nn.Sequential(*(list(pretrained_model.children())[:-1]))
        else:
            features = self._basic_cnn()
            output_dim = self.channel_sizes[-1]

        return features, output_dim

    def _basic_cnn(self):

        channel_sizes = [3] + self.channel_sizes
        kernel_sizes = (
            [self.kernel_sizes] * len(self.channel_sizes)
            if isinstance(self.kernel_sizes, int)
            else self.kernel_sizes
        )
        strides = (
            [self.strides] * len(self.channel_sizes)
            if isinstance(self.strides, int)
            else self.strides
        )

        BasicCNN = nn.Sequential()
        for i in range(1, len(channel_sizes)):
            BasicCNN.add_module(
                "conv_layer_{}".format(i - 1),
                conv_layer(
                    channel_sizes[i - 1],
                    channel_sizes[i],
                    kernel_sizes[i - 1],
                    strides[i - 1],
                    maxpool=i == 1,
                    adaptiveavgpool=i == len(channel_sizes) - 1,
                ),
            )
        return BasicCNN

    def _freeze(self, features):

        if self.trainable_params is not None:
            for name, param in features.named_parameters():
                for tl in trainable_params:
                    param.requires_grad = tl in name
        elif self.n_trainable is not None:
            for i, (name, param) in enumerate(
                reversed(list(features.named_parameters()))
            ):
                param.requires_grad = i < self.n_trainable
        else:
            warnings.warn(
                "Both 'trainable_params' and 'n_trainable' are 'None' and the entire network will be trained",
                UserWarning,
            )

    @staticmethod
    def get_backbone_output_dim(features):
        try:
            return features.fc.in_features
        except AttributeError:
            try:
                features.classifier.__dict__["_modules"]["0"].in_features
            except AttributeError:
                try:
                    return features.classifier.__dict__["_modules"]["1"].in_features
                except AttributeError:
                    return features.classifier.__dict__["_modules"]["1"].in_channels
