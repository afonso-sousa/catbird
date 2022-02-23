import argparse
import random
from pathlib import Path

import pandas as pd
from catbird.core import dump, load


def quora_data_prep(data_path, save_path=None, split_perc=0.3):
    """Prepare data related to Quora dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """
    data_path = Path(data_path)
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    quora_df = pd.read_csv(data_path / "quora_duplicate_questions.tsv", sep="\t")
    print(f"Original Quora dataset has {len(quora_df)} entries.")
    quora_df = quora_df[quora_df["is_duplicate"] == 1]
    print(f"After non-duplicate dropped, Quora dataset has {len(quora_df)} entries.")
    quora_df = quora_df.drop(["id", "qid1", "qid2", "is_duplicate"], axis=1)
    quora_df = quora_df.rename(columns={"question1": "src", "question2": "tgt"})
    quora = quora_df.to_dict("records")

    random.shuffle(quora)
    train_data = quora[: int(len(quora) * (1 - split_perc))]
    val_data = quora[int(len(quora) * (1 - split_perc)) :]

    train_filename = save_path / "quora_train.pkl"
    print(
        f"Quora train split ({len(train_data)} entries) is saved to '{train_filename}'"
    )
    dump(train_data, train_filename)
    val_filename = save_path / "quora_val.pkl"
    print(f"Quora val split ({len(val_data)} entries) is saved to '{val_filename}'")
    dump(val_data, val_filename)


def wikianswers_data_prep(data_path, save_path=None, split_perc=0.3):
    data_path = Path(data_path)
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    wikianswers_df = pd.read_csv(
        data_path / "questions.txt", sep="\t", header=None, on_bad_lines="skip"
    )
    print(f"Original WikiAnswers dataset has {len(wikianswers_df)} entries.")


def mscoco_data_prep(data_path, save_path=None, split="train"):
    data_path = Path(data_path)
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    
    file_dict = load(data_path / f"captions_{split}2014.json")

    # keys: ['info', 'images', 'licenses', 'annotations']
    annotations = file_dict["annotations"]
    print(f"Original MSCOCO dataset has {len(annotations)} entries.")

    # Get dict in form {id: List[references]}
    annotations_dict = {}
    for annot in annotations:
        annotations_dict.setdefault(annot["image_id"], []).append(annot["caption"])

    assert set(map(len, list(annotations_dict.values()))) == {5, 6, 7}

    mscoco = []
    for _, references in annotations_dict.items():
        random.shuffle(references)
        if len(references) % 2 != 0:
            del references[0]
        for i in range(0, len(references), 2):
            ref_dict = {"src": references[i], "tgt": references[i + 1]}
        mscoco.append(ref_dict)

    filename = save_path / f"mscoco_{split}.pkl"
    print(f"MSCOCO {split} split ({len(mscoco)} entries) is saved to '{filename}'")
    dump(mscoco, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Converter")
    parser.add_argument("dataset", help="name of the dataset")
    parser.add_argument(
        "--root-path",
        type=str,
        default="./data/quora",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./data/quora", help="specify path to store output"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="train or val split. Only used by some datasets."
    )
    args = parser.parse_args()

    if args.dataset == "quora":
        quora_data_prep(data_path=args.root_path, save_path=args.out_dir)
    elif args.dataset == "wikianswers":
        # wikianswers_data_prep(data_path=args.root_path, save_path=args.out_dir)
        print("In progress...")
    elif args.dataset == "mscoco":
        mscoco_data_prep(data_path=args.root_path, save_path=args.out_dir, split=args.split)

