# %%
from pathlib import Path
from unittest import skipIf
from datasets import load_dataset
from catbird.core import load

data_path = Path(__file__).parent.parent / "data"


@skipIf(
    not (data_path / "quora" / "quora_train.pkl").exists(),
    "train.txt is missing. Skipping 'test_quora_against_huggingface' test.",
)
def test_quora_against_huggingface():
    dataset = load_dataset("quora")
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["is_duplicate"] == True)
    
    train_data = load(data_path / "quora" / "quora_train.pkl")
    val_data = load(data_path / "quora" / "quora_val.pkl")
    quora = train_data + val_data
    t1 = sorted([[el["src"], el["tgt"]] for el in quora])
    t2 = sorted([dataset[i]["questions"]["text"] for i in range(len(dataset))])
    
    assert t1 == t2


# %%
