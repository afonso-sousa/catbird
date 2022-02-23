import argparse
from pathlib import Path

from catbird.core import dump, load
from stanza.server import CoreNLPClient
from tqdm import tqdm


def extract_triples(client, text):
    ann = client.annotate(text)
    triples = []
    for sentence in ann.sentence:
        for triple in sentence.openieTriple:
            triples.append(
                {
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                }
            )

    return triples


def is_object_substring(t, triples):
    object_substring_list = [
        t != u
        and t["relation"] == u["relation"]
        and all(w in u["object"] for w in t["object"].split())
        for u in triples
    ]
    return any(object_substring_list)


def disambiguate_triples(triples):
    return [t for t in triples if not is_object_substring(t, triples)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Extraction")
    parser.add_argument(
        "--root-path",
        type=str,
        default="./data/mscoco",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/mscoco",
        help="specify path to store output",
    )
    parser.add_argument(
        "--extract-for",
        type=str,
        default="src",
        choices=["src", "trg", "both"],
        help="number of threads to be used",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="train or val split."
    )
    args = parser.parse_args()

    root_path = Path(args.root_path)
    data = load(root_path / f"{root_path.name}_{args.split}.pkl")

    annotators = [
        "tokenize",
        "ssplit",
        "pos",
        "lemma",
        "depparse",
        "ner",
        "coref",
        "natlog",
        "openie",
    ]
    properties = {"openie.resolve_coref": True}

    triples_list = []
    with CoreNLPClient(
        annotators=annotators, properties=properties, be_quiet=True,
    ) as client:
        for entry in tqdm(data):
            triples = extract_triples(client, entry["src"])
            triples = disambiguate_triples(triples)
            triples_list.append(triples)

    filename = Path(args.out_dir) / f"mscoco_triples_{args.split}.pkl"
    print(
        f"MSCOCO IE triples {args.split} split ({len(triples_list)} entries) are saved to '{filename}'"
    )
    dump(triples_list, filename)
