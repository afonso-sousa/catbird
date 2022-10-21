# import torch


# def build_levi_graph(triples, tokenizer):
#     from torch_geometric.data import Data

#     # triples is list of dictionaries
#     nodes = set()
#     src_edges = []
#     trg_edges = []
#     for triple in triples:
#         nodes |= set(triple.values())
#         src_edges += [triple["subject"], triple["relation"]]
#         trg_edges += [triple["relation"], triple["object"]]

#     nodes = list(nodes)
#     src_edges = [nodes.index(edge) for edge in src_edges]
#     trg_edges = [nodes.index(edge) for edge in trg_edges]

#     if nodes:
#         nodes = tokenizer(nodes).input_ids

#     x = torch.tensor(nodes, dtype=torch.float)
#     edge_index = torch.tensor([src_edges, trg_edges], dtype=torch.long)

#     graph = Data(x=x, edge_index=edge_index)

#     return graph

import torch
from torch_geometric.data import Data


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

    special_tokens = [f"[{token}]" for token in all_pos + all_dep]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    x = torch.tensor(
        [[tokenizer.convert_tokens_to_ids(node)] for node in nodes], dtype=torch.float
    )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=x, edge_index=edge_index)

    return graph
