import inspect
from typing import Dict, Any


class Registry:
    """Creates a registration subgroup for modules to be added."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    def __repr__(self) -> str:
        format_str = (
            self.__class__.__name__
            + f"(name={self._name}, items={list(self._module_dict.keys())})"
        )
        return format_str

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict:
        return self._module_dict

    def get(self, key: str) -> Any:
        return self._module_dict.get(key, None)

    def _register_module(self, module_class: Any) -> None:
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, cls: Any) -> Any:
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry: Registry, default_args: dict = None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)
