import argparse
from pathlib import Path

from catbird.utils import Config, dump, load
from tqdm import tqdm

from sem_graph_utils import (
    build_tree,
    get_data_with_dependency_tree,
    get_graph,
    merge_corpus,
    prune,
    rearrange,
    tag,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build semantic graph")
    parser.add_argument("config", help="dataset config file path")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data",
        help="specify path to store output",
    )
    parser.add_argument(
        "--dp-file",
        type=str,
        help="dependency tree file",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    data = load(Path(cfg.data_root) / f"{cfg.dataset_name.lower()}_val.pkl")

    if args.dp_file:
        print(f"Loading dependency parsing information from {args.dp_file}.")
        dp_data = load(Path(args.dp_file))
    else:
        dp_data = get_data_with_dependency_tree(data)
        filename = Path(args.out_dir) / f"{cfg.dataset_name.lower()}_with_dp.pkl"
        print(f"Saving dependency parsing information to {filename}.")
        dump(dp_data, filename)

    graphs = []
    for sample in tqdm(dp_data, desc="Building Trees: ", total=len(dp_data)):
        sentence_graphs = []
        tree = build_tree(sample)
        pruned_tree = {
            "sequence": tree["words"],
            "tree": prune(tree["tree"], tree["words"]),
        }

        rearranged_tree = {
            "sequence": pruned_tree["sequence"],
            "tree": rearrange(pruned_tree["tree"], pruned_tree["sequence"]),
        }

        sentence_graphs.append(
            {
                "sequence": rearranged_tree["sequence"],
                "graph": get_graph(rearranged_tree["tree"]),
            }
        )

        graph = merge_corpus(sentence_graphs)
        graphs.append(graph)

        for idx, sample in tqdm(enumerate(graphs), desc="Tagging: ", total=len(graphs)):
            nodes, edges = sample["nodes"], sample["edges"]
            tgt_sentence = dp_data[idx]["tgt"]
            graphs[idx]["nodes"] = tag(nodes, edges, tgt_sentence)

    filename = Path(args.out_dir) / f"{cfg.dataset_name.lower()}_with_dp_processed.pkl"
    print(
        f"{cfg.dataset_name} with dependency tree ({len(graphs)} entries) is saved to '{filename}'"
    )
    dump(graphs, filename)
