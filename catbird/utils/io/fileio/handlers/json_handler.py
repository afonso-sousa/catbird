import json
from os import PathLike
from typing import IO, Any, Union

from .base import BaseFileHandler


class JsonHandler(BaseFileHandler):
    """JSON file handler."""

    def load_from_fileobj(self, file: IO[Any], **kwargs: Any) -> Any:
        return json.load(file, **kwargs)

    def load_from_path(self, filepath: Union[str, PathLike], **kwargs: Any) -> Any:
        with open(filepath, mode="r", encoding="utf-8") as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(
        self, obj: Any, filepath: Union[str, PathLike], **kwargs: Any
    ) -> None:
        with open(filepath, mode="w", encoding="utf-8") as f:
            self.dump_to_fileobj(obj, f, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: IO[Any], **kwargs: Any) -> None:
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> Union[str, bytes]:
        return json.dumps(obj, **kwargs)
