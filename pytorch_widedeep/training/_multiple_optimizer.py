from pytorch_widedeep.wdtypes import Dict, List, Union, Optimizer


class MultipleOptimizer(object):
    def __init__(self, opts: Dict[str, Union[Optimizer, List[Optimizer]]]) -> None:
        self._optimizers = opts

    def zero_grad(self):
        for _, op in self._optimizers.items():
            if isinstance(op, list):
                for _op in op:
                    _op.zero_grad()
            else:
                op.zero_grad()

    def step(self):
        for _, op in self._optimizers.items():
            if isinstance(op, list):
                for _op in op:
                    _op.step()
            else:
                op.step()
