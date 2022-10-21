from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool

from ..registry import GRAPH_ENCODERS


@GRAPH_ENCODERS.register_module
class GCNEncoder(nn.Module):
    def __init__(self, num_features, hidden_size, dropout=0):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # x = self.lin(x)  # [722, 50265]
        # print(x.shape)
        # q = F.linear(x, emb_weights)  # 512x50265
        # print(q.shape)

        x = global_add_pool(
            x, data.batch, size=data.num_graphs
        )  # batch_size x hidden_size

        # x = self.lin(x)
        # x, _ = to_dense_batch(x, data.batch)  # [batch_size, num_nodes, num_features]


        return x
