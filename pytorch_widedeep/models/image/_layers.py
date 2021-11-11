from torch import nn


def conv_layer(
    ni: int,
    nf: int,
    ks: int,
    stride: int,
    maxpool: bool,
    adaptiveavgpool: bool,
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
