"""File that defines Pickle file handler."""

import os
import pickle
from typing import Any, Union

from .base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    """Pickle file handler."""

    def load_from_fileobj(
        self, file: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> Any:
        """Load Python objects from Pickle file.

        Args:
            file (Union[str, bytes, os.PathLike]): Pickle file with a Python object representation.

        Returns:
            Any: returns the file-corresponding Python object.
        """
        return pickle.load(file, **kwargs)

    def load_from_path(
        self, filepath: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> Any:
        return super(PickleHandler, self).load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_fileobj(
        self, obj: Any, file: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> None:
        """Dump a Python object into a Pickle-formatted file.

        Args:
            obj (Any): a Python object to dump.
            file (Union[str, bytes, os.PathLike]): Pickle file with a Python object representation.
        """
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(
        self, obj: Any, filepath: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> None:
        super(PickleHandler, self).dump_to_path(obj, filepath, mode="wb", **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        """Serialize a Python object into a Pickle string format.

        Args:
            obj (Union[str, bytes, os.PathLike]): a Python object to dump into file.

        Returns:
            str: Pickle string-formatted serialized object.
        """
        return pickle.dumps(obj, **kwargs)
