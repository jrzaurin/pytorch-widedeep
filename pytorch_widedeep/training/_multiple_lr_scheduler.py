from pytorch_widedeep.wdtypes import Dict, LRScheduler


class MultipleLRScheduler(object):
    def __init__(self, scheds: Dict[str, LRScheduler]):
        self._schedulers = scheds

    def step(self):
        for _, sc in self._schedulers.items():
            sc.step()
