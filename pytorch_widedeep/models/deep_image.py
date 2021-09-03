from torch import nn
from torchvision import models

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP


def conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    maxpool: bool = True,
    adaptiveavgpool: bool = False,
):
    layer = nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=True, stride=stride, padding=ks // 2),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )
    if maxpool:
        layer.add_module("maxpool", nn.MaxPool2d(2, 2))
    if adaptiveavgpool:
        layer.add_module("adaptiveavgpool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layer


class DeepImage(nn.Module):
    r"""Standard image classifier/regressor using a pretrained network (in
    particular ResNets) or a sequence of 4 convolution layers.

    If ``pretrained=False`` the `'backbone'` of :obj:`DeepImage` will be a
    sequence of 4 convolutional layers comprised by: ``Conv2d -> BatchNorm2d
    -> LeakyReLU``. The 4th one will also add a final ``AdaptiveAvgPool2d``
    operation.

    If ``pretrained=True`` the `'backbone'` will be ResNets. ResNets have
    9 `'components'` before the last dense layers. The first 4 are:
    ``Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d``. Then there are 4
    additional resnet blocks comprised by a series of convolutions and
    then the final ``AdaptiveAvgPool2d``. Overall, ``4+4+1=9``. The
    parameter ``freeze_n`` sets the number of layers to be frozen. For
    example, ``freeze_n=6`` will freeze all but the last 3 layers.

    In addition to all of the above, there is the option to add a fully
    connected set of dense layers (referred as `imagehead`) on top of the
    stack of CNNs

    Parameters
    ----------
    pretrained: bool, default = True
        Indicates whether or not we use a pretrained Resnet network or a
        series of conv layers (see conv_layer function)
    resnet_architecture: int, default = 18
        The resnet architecture. One of 18, 34 or 50
    freeze_n: int, default = 6
        number of layers to freeze. Must be less than or equal to 8. If 8
        the entire 'backbone' of the network will be frozen
    head_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the head. e.g: [64,32]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        ``tanh``, ``relu``, ``leaky_relu`` and ``gelu`` are supported
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
    backbone: ``nn.Sequential``
        Sequential stack of CNNs comprising the 'backbone' of the network
    imagehead: ``nn.Sequential``
        Sequential stack of dense layers comprising the FC-Head (aka imagehead)
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepImage
    >>> X_img = torch.rand((2,3,224,224))
    >>> model = DeepImage(head_hidden_dims=[512, 64, 8])
    >>> out = model(X_img)
    """

    def __init__(
        self,
        pretrained: bool = True,
        resnet_architecture: int = 18,
        freeze_n: int = 6,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: float = 0.1,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
    ):
        super(DeepImage, self).__init__()

        self.pretrained = pretrained
        self.resnet_architecture = resnet_architecture
        self.freeze_n = freeze_n
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first

        if pretrained:
            vision_model = self.select_resnet_architecture(resnet_architecture)
            backbone_layers = list(vision_model.children())[:-1]
            self.backbone = self._build_backbone(backbone_layers, freeze_n)
        else:
            self.backbone = self._conv_nn()

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = 512

        if self.head_hidden_dims is not None:
            assert self.head_hidden_dims[0] == self.output_dim, (
                "The output dimension from the backbone ({}) is not consistent with "
                "the expected input dimension ({}) of the fc-head".format(
                    self.output_dim, self.head_hidden_dims[0]
                )
            )
            self.imagehead = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
            self.output_dim = head_hidden_dims[-1]

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass connecting the `'backbone'` with the `'head layers'`"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if self.head_hidden_dims is not None:
            out = self.imagehead(x)
            return out
        else:
            return x

    @staticmethod
    def select_resnet_architecture(resnet_architecture: int):
        if resnet_architecture == 18:
            return models.resnet18(pretrained=True)
        elif resnet_architecture == 34:
            return models.resnet34(pretrained=True)
        elif resnet_architecture == 50:
            return models.resnet50(pretrained=True)

    def _conv_nn(self):
        return nn.Sequential(
            conv_layer(3, 64, 3),
            conv_layer(64, 128, 1, maxpool=False),
            conv_layer(128, 256, 1, maxpool=False),
            conv_layer(256, 512, 1, maxpool=False, adaptiveavgpool=True),
        )

    def _build_backbone(self, backbone_layers, freeze_n):
        """
        Builds the backbone layers
        """
        if freeze_n > 8:
            raise ValueError(
                "freeze_n' must be less than or equal to 8 for resnet architectures"
            )
        frozen_layers = []
        trainable_layers = backbone_layers[freeze_n:]
        for layer in backbone_layers[:freeze_n]:
            for param in layer.parameters():
                param.requires_grad = False
            frozen_layers.append(layer)
        trainable_and_frozen_layers = frozen_layers + trainable_layers
        return nn.Sequential(*trainable_and_frozen_layers)
