from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool

from ..registry import GRAPH_ENCODERS

@GRAPH_ENCODERS.register_module
class GCNEncoder(nn.Module):
    def __init__(self,
                 num_features,
                 hidden_size,
                 dropout=0):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((32, hidden_size))

    def forward(self, data):
        # batch_size = data.num_graphs
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = global_add_pool(x, data.batch, size=data.num_graphs)

        return x