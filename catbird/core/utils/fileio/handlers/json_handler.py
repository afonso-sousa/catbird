"""File that defines JSON file handler."""

import json
import os
from typing import Any, Union

from .base import BaseFileHandler


class JsonHandler(BaseFileHandler):
    """JSON file handler."""

    def load_from_fileobj(self, file: Union[str, bytes, os.PathLike]) -> Any:
        """Load Python objects from JSON file.

        Args:
            file (Union[str, bytes, os.PathLike]): JSON file with a Python object representation.

        Returns:
            Any: returns the file-corresponding Python object.
        """
        return json.load(file)

    def dump_to_fileobj(
        self, obj: Any, file: Union[str, bytes, os.PathLike], **kwargs: Any
    ) -> None:
        """Dump a Python object into a JSON-formatted file.

        Args:
            obj (Any): a Python object to dump.
            file (Union[str, bytes, os.PathLike]): JSON file with a Python object representation.
        """
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> str:
        """Serialize a Python object into a JSON string format.

        Args:
            obj (Union[str, bytes, os.PathLike]): a Python object to dump into file.

        Returns:
            str: JSON string-formatted serialized object.
        """
        return json.dumps(obj, **kwargs)
