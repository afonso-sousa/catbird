from pathlib import Path
from unittest import skipIf

from catbird.core import load
from datasets import load_dataset, concatenate_datasets

data_path = Path(__file__).parent.parent / "data"


@skipIf(
    not (data_path / "quora" / "quora_train.pkl").exists(),
    "train.txt is missing. Skipping 'test_quora_against_huggingface' test.",
)
def test_quora_against_huggingface():
    dataset = load_dataset("quora")
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["is_duplicate"] == True)
    dataset = dataset.train_test_split(test_size=0.3) # split percentage hardcoded

    train_data = load(data_path / "quora" / "quora_train.pkl")
    val_data = load(data_path / "quora" / "quora_val.pkl")

    assert len(train_data) == len(dataset["train"])
    assert len(val_data) == len(dataset["test"])

    my_quora = train_data + val_data
    hf_quora = concatenate_datasets([dataset['train'], dataset['test']])
    
    t1 = sorted([[el["src"], el["tgt"]] for el in my_quora])
    t2 = sorted([hf_quora[i]["questions"]["text"] for i in range(len(hf_quora))])
    assert t1 == t2

