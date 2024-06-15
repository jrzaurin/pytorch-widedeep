from pytorch_widedeep.wdtypes import Dict, List, Union, LRScheduler


class MultipleLRScheduler(object):
    def __init__(
        self, scheds: Dict[str, Union[LRScheduler, List[LRScheduler]]]
    ) -> None:
        self._schedulers = scheds

    def step(self):
        for _, sc in self._schedulers.items():
            if isinstance(sc, list):
                for _sc in sc:
                    _sc.step()
            else:
                sc.step()
