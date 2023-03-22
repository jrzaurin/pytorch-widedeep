from torch import nn


class BaseWDModelComponent(nn.Module):
    @property
    def output_dim(self) -> int:
        return NotImplementedError(  # type: ignore[return-value]
            "All models passed to the WideDeep class must contain an 'output_dim' "
            "property or attribute. This is the dimension of the output tensor coming "
            "from the backbone model that will be connected to the final prediction "
            "layer or fully connected head"
        )
