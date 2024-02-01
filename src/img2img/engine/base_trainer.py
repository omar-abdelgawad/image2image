from typing import Protocol
from typing import Any


class Trainer(Protocol):
    def train(self, *args, **kwargs) -> Any:
        ...

    def eval(self, *args, **kwargs) -> Any:
        ...

    def sample(self, *args, **kwargs) -> Any:
        ...
