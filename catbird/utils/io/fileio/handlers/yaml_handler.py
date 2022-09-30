"""File that defines YAML file handler."""

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper  # type: ignore

import os
from typing import Any, Union

from .base import BaseFileHandler


class YamlHandler(BaseFileHandler):
    """YAML file handler."""

    def load_from_fileobj(
        self, file: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> Any:
        """Load Python objects from YAML file.

        Args:
            file (Union[str, bytes, os.PathLike]): YAML file with a Python object representation.

        Returns:
            Any: returns the file-corresponding Python object.
        """
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(
        self, obj: Any, file: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> None:
        """Dump a Python object into a YAML-formatted file.

        Args:
            obj (Any): a Python object to dump.
            file (Union[str, bytes, os.PathLike]): YAML file with a Python object representation.
        """
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        """Serialize a Python object into a YAML string format.

        Args:
            obj (Union[str, bytes, os.PathLike]): a Python object to dump into file.

        Returns:
            str: YAML string-formatted serialized object.
        """
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
