from torch import nn
from torchvision import models

from ..wdtypes import *
from .deep_dense import dense_layer


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
    r"""
    Standard image classifier/regressor using a pretrained network (in
    particular ResNets) or a sequence of 4 convolution layers.

    If ``pretrained=False`` the `'backbone'` of :obj:`DeepImage` will be a
    sequence of 4 convolutional layers comprised by: ``Conv2d -> BatchNorm2d
    -> LeakyReLU``. The 4th one will also add a final ``AdaptiveAvgPool2d``
    operation.

    If ``pretrained=True`` the `'backbone'` will be ResNets. ResNets have 9
    `'components'` before the last dense layers. The first 4 are: ``Conv2d ->
    BatchNorm2d -> ReLU -> MaxPool2d``. Then there are 4 additional resnet
    blocks comprised by a series of convolutions and then the final
    ``AdaptiveAvgPool2d``. Overall, ``4+4+1=9``. The parameter ``freeze`` sets
    the layers to be frozen. For example, ``freeze=6`` will freeze all but the
    last 3 layers. If ``freeze=all`` the entire network will be frozen.

    In addition to all of the above, there is the option to add a fully
    connected set of dense layers (referred as `imagehead`) on top of the
    stack of RNNs

    Parameters
    ----------
    pretrained: bool
        Indicates whether or not we use a pretrained Resnet network or a
        series of conv layers (see conv_layer function)
    resnet: int
        The resnet architecture. One of 18, 34 or 50
    freeze: Union[str, int]
        number of layers to freeze. If int must be less than 8. The only
        string allowed is 'all' which will freeze the entire network
    head_layers: List, Optional
        List with the sizes of the stacked dense layers in the head
        e.g: [128, 64]
    head_dropout: List, Optional
        List with the dropout between the dense layers. e.g: [0.5, 0.5].
    head_batchnorm: bool, Optional
        Boolean indicating whether or not to include batch normalizatin in the
        dense layers that form the imagehead

    Attributes
    ----------
    backbone: :obj:`nn.Sequential`
        Sequential stack of CNNs comprising the 'backbone' of the network
    imagehead: :obj:`nn.Sequential`
        Sequential stack of dense layers comprising the FC-Head (aka imagehead)
    output_dim: :obj:`int`
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepImage
    >>> X_img = torch.rand((2,3,224,224))
    >>> model = DeepImage(head_layers=[512, 64, 8])
    >>> model(X_img)
    tensor([[ 7.7234e-02,  8.0923e-02,  2.3077e-01, -5.1122e-03, -4.3018e-03,
              3.1193e-01,  3.0780e-01,  6.5098e-01],
            [ 4.6191e-02,  6.7856e-02, -3.0163e-04, -3.7670e-03, -2.1437e-03,
              1.5416e-01,  3.9227e-01,  5.5048e-01]], grad_fn=<LeakyReluBackward1>)
    """

    def __init__(
        self,
        pretrained: bool = True,
        resnet: int = 18,
        freeze: Union[str, int] = 6,
        head_layers: Optional[List[int]] = None,
        head_dropout: Optional[List[float]] = None,
        head_batchnorm: Optional[bool] = False,
    ):
        super(DeepImage, self).__init__()

        self.head_layers = head_layers

        if pretrained:
            if resnet == 18:
                vision_model = models.resnet18(pretrained=True)
            elif resnet == 34:
                vision_model = models.resnet34(pretrained=True)
            elif resnet == 50:
                vision_model = models.resnet50(pretrained=True)

            backbone_layers = list(vision_model.children())[:-1]

            if isinstance(freeze, str):
                frozen_layers = []
                for layer in backbone_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)
                self.backbone = nn.Sequential(*frozen_layers)
            if isinstance(freeze, int):
                assert (
                    freeze < 8
                ), "freeze' must be less than 8 when using resnet architectures"
                frozen_layers = []
                trainable_layers = backbone_layers[freeze:]
                for layer in backbone_layers[:freeze]:
                    for param in layer.parameters():
                        param.requires_grad = False
                    frozen_layers.append(layer)

                backbone_layers = frozen_layers + trainable_layers
                self.backbone = nn.Sequential(*backbone_layers)
        else:
            self.backbone = nn.Sequential(
                conv_layer(3, 64, 3),
                conv_layer(64, 128, 1, maxpool=False),
                conv_layer(128, 256, 1, maxpool=False),
                conv_layer(256, 512, 1, maxpool=False, adaptiveavgpool=True),
            )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = 512

        if self.head_layers is not None:
            assert self.head_layers[0] == self.output_dim, (
                "The output dimension from the backbone ({}) is not consistent with "
                "the expected input dimension ({}) of the fc-head".format(
                    self.output_dim, self.head_layers[0]
                )
            )
            if not head_dropout:
                head_dropout = [0.0] * len(head_layers)
            self.imagehead = nn.Sequential()
            for i in range(1, len(head_layers)):
                self.imagehead.add_module(
                    "dense_layer_{}".format(i - 1),
                    dense_layer(
                        head_layers[i - 1],
                        head_layers[i],
                        head_dropout[i - 1],
                        head_batchnorm,
                    ),
                )
            self.output_dim = head_layers[-1]

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass connecting the `'backbone'` with the `'head layers'`
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if self.head_layers is not None:
            out = self.imagehead(x)
            return out
        else:
            return x
