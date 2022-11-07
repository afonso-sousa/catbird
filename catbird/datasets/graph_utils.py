import torch
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from torch import nn


def build_dependency_structure(sample, all_dep, all_pos, tokenizer):
    nodes = [[token["word"], token["pos"]] for token in sample["src_dp"]]

    edges = []
    edge_attr = []
    for idx, token in enumerate(sample["src_dp"]):
        if token["head"] != -1:
            edges.append(
                [
                    token["head"],
                    idx,
                ]
            )
            edge_attr.append(token["dep"])

    special_tokens_dict = {"additional_special_tokens": all_pos + all_dep}
    tokenizer.add_special_tokens(special_tokens_dict)

    x = torch.tensor(
        [tokenizer.convert_tokens_to_ids(node) for node in nodes], dtype=torch.float
    )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_type = torch.tensor(
        [tokenizer.convert_tokens_to_ids(attr) for attr in edge_attr], dtype=torch.long
    )

    graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    return graph


def build_levi_graph(sample, all_dep, all_pos, tokenizer):
    from torch_geometric.data import Data

    nodes = [token["word"] for token in sample["src_dp"]]

    edges = []
    for idx, token in enumerate(sample["src_dp"]):
        nodes.append(f'[{token["dep"]}]')
        if token["head"] != -1:
            edges.append(
                [
                    token["head"],
                    nodes.index(f'[{token["dep"]}]'),
                ]
            )
            edges.append(
                [
                    nodes.index(f'[{token["dep"]}]'),
                    idx,
                ]
            )

    # special_tokens = [f"[{token}]" for token in all_pos + all_dep]
    # special_tokens_dict = {"additional_special_tokens": special_tokens}
    # tokenizer.add_special_tokens(special_tokens_dict)

    x = torch.tensor(
        [[tokenizer.convert_tokens_to_ids(node)] for node in nodes], dtype=torch.float
    )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=x, edge_index=edge_index)

    return graph


def build_graph(sample, all_dep, all_pos, tokenizer):
    from torch_geometric.data import Data

    max_len = 100

    nodes = [token["word"] for token in sample["src_dp"]]

    edges = []
    edge_type = []
    for idx, token in enumerate(sample["src_dp"]):
        if token["head"] != -1:
            edges.append(
                [
                    token["head"],
                    idx,
                ]
            )
        else:
            edges.append(
                [
                    idx,
                    idx,
                ]
            )
        edge_type.append(all_dep.index(token["dep"]))

    x = [torch.tensor(tokenizer(node).input_ids[1:-1]) for node in nodes]

    # pad first seq to desired length
    x[0] = nn.ConstantPad1d((0, max_len - x[0].shape[0]), -1)(x[0])
    x = pad_sequence(x, batch_first=True, padding_value=-1)

    x = x.type(torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type)

    graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    return graph
    # return {"nodes": x, "edge_index": edge_index, "edge_types": edge_types}
