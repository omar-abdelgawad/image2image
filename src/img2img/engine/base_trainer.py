from typing import Protocol
from typing import Any


class Trainer(Protocol):
    def train(self, *args: Any, **kwargs: Any) -> Any: ...

    def eval(self, *args: Any, **kwargs: Any) -> Any: ...

    def sample(self, *args: Any, **kwargs: Any) -> Any: ...
