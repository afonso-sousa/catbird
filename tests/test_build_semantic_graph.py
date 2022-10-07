from pathlib import Path
from unittest import skipIf

import pytest
from catbird.core import load
from tools.preprocessing.sem_graph_utils import get_data_with_dependency_tree

data_path = Path(__file__).parent.parent / "data"


@skipIf(
    not (data_path / "quora" / "quora_train.pkl").exists(),
    "train.txt is missing. Skipping 'test_quora_against_huggingface' test.",
)
def test_build_dependency_parsing():
    data = load(data_path / "quora" / "quora_val.pkl")
    try:
        get_data_with_dependency_tree(data[:5])
    except:
        pytest.fail("Something went wrong when trying to semantic parse.")
