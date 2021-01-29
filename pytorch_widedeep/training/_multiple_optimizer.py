from pytorch_widedeep.wdtypes import *  # noqa: F403


class MultipleOptimizer(object):
    def __init__(self, opts: Dict[str, Optimizer]):
        self._optimizers = opts

    def zero_grad(self):
        for _, op in self._optimizers.items():
            op.zero_grad()

    def step(self):
        for _, op in self._optimizers.items():
            op.step()
