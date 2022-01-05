import argparse
import random
from pathlib import Path

import pandas as pd
from catbird.core import dump


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
    quora_df = quora_df[quora_df["is_duplicate"] == 1]
    quora_df = quora_df.drop(["id", "qid1", "qid2", "is_duplicate"], axis=1)
    quora_df = quora_df.rename(columns={"question1": "src", "question2": "tgt"})
    quora = quora_df.to_dict("records")

    random.shuffle(quora)
    train_data = quora[: int(len(quora) * (1 - split_perc))]
    val_data = quora[int(len(quora) * (1 - split_perc)) :]

    train_filename = save_path / "quora_train.pkl"
    print(f"Quora train split is saved to '{train_filename}'")
    dump(train_data, train_filename)
    val_filename = save_path / "quora_val.pkl"
    print(f"Quora val split is saved to '{val_filename}'")
    dump(val_data, val_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("dataset", help="name of the dataset")
    parser.add_argument(
        "--root-path",
        type=str,
        default="./data/quora",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./data/kitti", help="name of info pkl"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    args = parser.parse_args()

    if args.dataset == "quora":
        quora_data_prep(data_path=args.root_path, save_path=args.out_dir)

