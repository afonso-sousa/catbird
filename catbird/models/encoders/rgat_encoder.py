from torch import nn
from torch_geometric.nn import RGATConv, global_add_pool
from torch_geometric.utils import to_dense_batch
import torch

from ..registry import GRAPH_ENCODERS


@GRAPH_ENCODERS.register_module
class RGATEncoder(nn.Module):
    def __init__(self, num_relations, in_channels, hidden_size, dropout=0):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_size, num_relations)
        self.conv2 = RGATConv(hidden_size, hidden_size, num_relations)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, data, embedding_weights):
        x_with_pad, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # each entry of x_with_pad is shaped as 1 x 100, with integers according with tokenizer and -1 as padding
        x = x_with_pad.new_zeros(x_with_pad.shape[0], embedding_weights.shape[1])
        for i, node in enumerate(x_with_pad):
            indices = node[node > -1].type(torch.long)
            node_features = embedding_weights[indices].mean(dim=0)
            x[i] = node_features

        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.lin(x)
        # x_nodes, _ = to_dense_batch(
        #     x, data.batch
        # )  # [batch_size, num_nodes, num_features]

        # input_ids = input_ids[1:-1] # remove bos and eos
        # for idx_in_batch, sentence in enumerate(input_ids):
        #     for idx_in_sentence, id in enumerate(sentence):
        #         embedding_weights[id] = torch.mean(embedding_weights[id], x_nodes[idx_in_batch, idx_in_sentence, :])
        # return embedding_weights

        x = global_add_pool(
            x, data.batch, size=data.num_graphs
        )  # batch_size x num_features

        return x
