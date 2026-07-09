"""Dataset DataModules. Register custom datasets with @data.register("name")."""

from rfsensing.registry import Registry

_registry = Registry("dataset")
register = _registry.register
build = _registry.build
list_available = _registry.available

from rfsensing.data import synthetic, ut_har  # noqa: E402, F401
from rfsensing.data.base import CSIDataModule  # noqa: E402, F401