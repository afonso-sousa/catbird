from abc import ABCMeta, abstractmethod
from typing import Any, IO, Union
from os import PathLike


class BaseFileHandler:
    """Base abstract class for file handlers."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def load_from_fileobj(self, file: IO[Any], **kwargs: Any) -> Any:
        """Abstraction for loading Python objects from files.

        Args:
            file (IO[Any]): file with a Python object representation.

        Returns:
            Any: returns the file-corresponding Python object.
        """
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj: Any, file: IO[Any], **kwargs: Any) -> None:
        """Abstraction for dumping a Python object into a file format.

        Args:
            obj (Any): a Python object to dump into file.
            file (IO[Any]): file with a Python object representation.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj: Any, **kwargs: Any) -> Union[str, bytes]:
        """Abstraction for serializing a Python object into a string format.

        Args:
            obj (Any): a Python object to dump into file.

        Returns:
            str: String-formatted serialized object.
        """
        pass

    @abstractmethod
    def load_from_path(self, filepath: Union[str, PathLike], **kwargs: Any) -> Any:
        """Load serialized object from file.

        Args:
            filepath (Union[str, Path]): Path to a file.
            mode (str, optional): Specifies the mode in which the file is opened. Defaults to "r".

        Returns:
            Any: a Python object.
        """
        pass

    @abstractmethod
    def dump_to_path(self, obj: Any, filepath: Union[str, PathLike], **kwargs) -> None:
        """Dump a serialized object into a file.

        Args:
            obj (Any): a Python object
            filepath (Union[str, Path]): Path to a file.
            mode (str, optional): Specifies the mode in which the file is opened. Defaults to "w".
        """
        pass
