from torch import nn
from torch_geometric.nn import RGATConv
from torch_geometric.utils import to_dense_batch

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

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.lin(x)
        x, _ = to_dense_batch(x, data.batch)  # [batch_size, num_nodes, num_features]

        # x = global_add_pool(x, data.batch, size=data.num_graphs)

        return x
