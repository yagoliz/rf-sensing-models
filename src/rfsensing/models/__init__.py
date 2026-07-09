"""Model zoo. Register custom architectures with @models.register("name")."""

from rfsensing.registry import Registry

_registry = Registry("model")
register = _registry.register
build = _registry.build
list_available = _registry.available

# Import submodules so their @register decorators run.
from rfsensing.models import lenet, mlp, resnet, rnn  # noqa: E402, F401