"""Generic name -> factory registry used for models and datasets."""

from collections.abc import Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def decorator(factory: Callable) -> Callable:
            if name in self._factories:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs):
        try:
            factory = self._factories[name]
        except KeyError:
            available = ", ".join(self.available()) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            ) from None
        return factory(**kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)
