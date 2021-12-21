"""File with configuration classes."""

import os
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from addict import Dict  # type: ignore

from .fileio import dump as f_dump
from .fileio import load
from .path import check_file_exist


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: json/yaml.
    The interface is the same as a dict object and also allows access config values as attributes.
    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/torchie/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/torchie/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    def __init__(self, cfg_dict: dict = None, filename: str = None) -> None:
        """Init config parameters.

        Args:
            cfg_dict (dict, optional): Dictionary to init configuration. Defaults to None.
            filename (str, optional): Filename from which to init configuration. Defaults to None.

        Raises:
            TypeError: Raises exception if cfg_dict is not a dictionary
        """
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", Dict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if filename:
            with open(filename, "r") as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    @staticmethod
    def fromfile(filename: str) -> "Config":
        """Create a Config instance with the information from a configuration file.

        Args:
            filename (str): Name of file to load information from.

        Raises:
            IOError: The specified file's type is not supported.

        Returns:
            Config: a config instance with the information in the provided file.
        """
        filename = Path(filename).resolve()
        check_file_exist(filename)
        if filename.suffix in [".yml", ".yaml", ".json"]:
            cfg_dict = load(filename)
        else:
            raise IOError("Only yml/yaml/json types are supported!")
        return Config(cfg_dict, filename=filename)

    def dump(
        self, file: Optional[Union[str, bytes, os.PathLike]] = None
    ) -> Optional[str]:
        """Dump the information in a Config instance into a file.

        Args:
            file (Optional[TextIO], optional): File to dump information to. Defaults to None.

        Returns:
            [Optional[str], optional]: A string representation of the information in the Config instance or nothing.
        """
        cfg_dict = super(Config, self).__getattribute__("_cfg_dict").to_dict()
        if file is None:
            file_format = self.filename.split(".")[-1]
            return f_dump(cfg_dict, file_format=file_format)
        else:
            f_dump(cfg_dict, file)

    @property
    def filename(self) -> Optional[str]:
        """Get filename if set.

        Returns:
            Optional[str]: File name or nothing.
        """
        return self._filename

    @property
    def text(self) -> str:
        """Get text.

        Returns:
            str: File text or empty string.
        """
        return self._text

    def __repr__(self) -> str:
        """Printable representation of a Config object.

        Returns:
            str: Object representation.
        """
        return "Config (path: {}): {}".format(self.filename, self._cfg_dict.__repr__())

    def __len__(self) -> int:
        """Get number of entries in Config object.

        Returns:
            int: Length of Config instance.
        """
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        """Get a named attribute from a Config instance with function syntax.

        Args:
            name (str): Name of attribute.

        Returns:
            Any: Value for given attribute name.
        """
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str) -> Any:
        """Get a named attribute from a Config instance with dictionary syntax.

        Args:
            name (str): Name of attribute.
        Returns:
            Any: Value for given attribute name.
        """
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a named attribute to a Config instance with function syntax.

        Args:
            name (str): Name of attribute.
            value (Any): Value to set.
        """
        if isinstance(value, dict):
            value = Dict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        """Set a named attribute to a Config instance with dictionary syntax.

        Args:
            name (str): Name of attribute.
            value (Any): Value to set.
        """
        if isinstance(value, dict):
            value = Dict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self) -> Iterator:
        """Get an iterator version of the Config object.

        Returns:
            [type]: an iterator version of the Config object.
        """
        return iter(self._cfg_dict)
