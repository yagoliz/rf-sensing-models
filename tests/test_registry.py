import pytest

from rfsensing.registry import Registry


def _make_registry():
    reg = Registry("widget")

    @reg.register("box")
    class Box:
        def __init__(self, size=1):
            self.size = size

    return reg, Box


def test_build_returns_instance_with_kwargs():
    reg, Box = _make_registry()
    obj = reg.build("box", size=3)
    assert isinstance(obj, Box)
    assert obj.size == 3


def test_available_is_sorted():
    reg, _ = _make_registry()

    @reg.register("axe")
    class Axe:
        pass

    assert reg.available() == ["axe", "box"]


def test_unknown_name_raises_and_lists_available():
    reg, _ = _make_registry()
    with pytest.raises(KeyError, match=r"Unknown widget 'nope'.*box"):
        reg.build("nope")


def test_duplicate_registration_raises():
    reg, _ = _make_registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register("box")(object)