from torch import nn


def conv_layer(
    ni: int,
    nf: int,
    kernel_size: int = 3,
    stride: int = 1,
    maxpool: bool = True,
    adaptiveavgpool: bool = False,
):
    layer = nn.Sequential(
        nn.Conv2d(
            ni,
            nf,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=kernel_size // 2,
        ),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )
    if maxpool:
        layer.add_module("maxpool", nn.MaxPool2d(2, 2))
    if adaptiveavgpool:
        layer.add_module("adaptiveavgpool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layer
