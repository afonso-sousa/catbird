"""File that defines base abstract class for file handlers."""

import os
from abc import ABCMeta, abstractmethod
from typing import Any, IO


class BaseFileHandler:
    """Base abstract class for file handlers."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def load_from_fileobj(self, file: IO, **kwargs: Any) -> Any:
        """Abstraction for loading Python objects from files.

        Args:
            file (Union[str, bytes, os.PathLike]): file with a Python object representation.

        Returns:
            Any: returns the file-corresponding Python object.
        """
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj: Any, file: IO, **kwargs: Any) -> None:
        """Abstraction for dumping a Python object into a file format.

        Args:
            obj (Any): a Python object to dump into file.
            file (Union[str, bytes, os.PathLike]): file with a Python object representation.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        """Abstraction for serializing a Python object into a string format.

        Args:
            obj (Any): a Python object to dump into file.

        Returns:
            str: String-formatted serialized object.
        """
        pass

    def load_from_path(self, filepath: IO, mode: str = "r", **kwargs: Any) -> Any:
        """Load serialized object from file.

        Args:
            filepath (Union[str, bytes, os.PathLike]): Path to a file.
            mode (str, optional): Specifies the mode in which the file is opened. Defaults to "r".

        Returns:
            Any: a Python object.
        """
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj: Any, filepath: IO, mode: str = "w", **kwargs) -> None:
        """Dump a serialized object into a file.

        Args:
            obj (Any): a Python object
            filepath (Union[str, bytes, os.PathLike]): Path to a file.
            mode (str, optional): Specifies the mode in which the file is opened. Defaults to "w".
        """
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
