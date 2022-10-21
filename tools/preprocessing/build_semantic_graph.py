import argparse
from pathlib import Path
import random
import spacy

from catbird.utils import Config, dump, load, fopen
from tqdm import tqdm
from spacy import displacy

from sem_graph_utils import (
    build_tree,
    get_data_with_dependency_tree,
    get_graph,
    merge_corpus,
    prune,
    rearrange,
    tag,
)


def main(args, cfg, dp_data):
    if args.just_dependency_structure:
        return

    pipeline = spacy.load("en_core_web_sm")

    if args.display_sample:
        sampled_sentence = random.choice(dp_data)["src"]
        svg_img = displacy.render(
            pipeline(sampled_sentence), style="dep", options={"distance": 120}
        )
        output_path = Path("dependency_plot.svg")
        with fopen(output_path, mode="w", encoding="utf-8") as f:
            f.write(svg_img)

    graphs = []
    for sample in tqdm(dp_data, desc="Building Trees: ", total=len(dp_data)):
        sentence_graphs = []
        dep_parse = sample["src_dp"]
        tokens = [token.text for token in pipeline(sample["src"])]

        tree = build_tree(dep_parse, tokens)
        pruned_tree = prune(tree, tokens)
        rearranged_tree = rearrange(pruned_tree, tokens)
        sentence_graphs.append(
            {
                "sequence": tokens,
                "graph": get_graph(rearranged_tree),
            }
        )

        graph = merge_corpus(sentence_graphs)
        graphs.append(graph)

        for idx, sample in enumerate(graphs):
            nodes, edges = sample["nodes"], sample["edges"]
            tgt_sentence = dp_data[idx]["tgt"]
            graphs[idx]["nodes"] = tag(nodes, edges, tgt_sentence)

    filename = Path(cfg.data_root) / f"{cfg.dataset_name.lower()}_with_dp_processed.pkl"
    print(
        f"{cfg.dataset_name} with dependency tree ({len(graphs)} entries) is saved to '{filename}'"
    )
    dump(graphs, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build semantic graph")
    parser.add_argument("config", help="dataset config file path")
    # parser.add_argument(
    #     "--out-dir",
    #     type=str,
    #     default="./data",
    #     help="specify path to store output",
    # )
    group_generate_or_exists = parser.add_mutually_exclusive_group()
    group_generate_or_exists.add_argument(
        "--dp-file",
        type=str,
        help="dependency tree file",
    )
    group_generate_or_exists.add_argument(
        "--just-dependency-structure",
        action="store_true",
        help="whether to get just the raw dependency structure",
    )
    parser.add_argument(
        "--display-sample",
        action="store_true",
        help="whether to display a randomly sampled dependency graphs",
    )
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="train or val split",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    data = load(Path(cfg.data_root) / f"{cfg.dataset_name.lower()}_{args.split}.pkl")

    if args.dp_file:
        print(f"Loading dependency parsing information from {args.dp_file}.")
        dp_data = load(Path(args.dp_file))
    else:
        dp_data = get_data_with_dependency_tree(data)
        filename = (
            Path(cfg.data_root) / f"{cfg.dataset_name.lower()}_with_dp_{args.split}.pkl"
        )
        print(f"Saving dependency parsing information to {filename}.")
        dump(dp_data, filename)

    main(args, cfg, dp_data)


# %%
import os

os.chdir(os.path.join(os.getcwd(), "../.."))

# %%
from catbird.utils import Config, dump, load, fopen
from pathlib import Path

data = load(Path("data/quora/quora_with_dp_val.pkl"))
# %%
