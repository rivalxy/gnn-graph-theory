import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.data
from torch_geometric.nn import GINConv, GPSConv, global_add_pool


class GIN(nn.Module):
    """
    GIN: "How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019)
    https://arxiv.org/abs/1810.00826
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            # Paper Sec 3: MLP with BN between layers (not just a single linear)
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Paper Sec 4.2: graph-level readout uses all layers, not just the last
        # Each layer contributes a pooled graph embedding -> sum them
        # Final MLP head operates on hidden_dim (post-sum)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Paper Eq. 4.2: collect graph-level readout from every layer
        h_graph = 0
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            x = F.dropout(x, self.dropout, training=self.training)
            h_graph = h_graph + global_add_pool(x, batch)  # sum over layers

        # Paper Sec 4.2: MLP classifier on the summed representation
        h = F.relu(self.lin1(h_graph))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.classifier(h).view(-1)


class GPS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attn_dropout: float,
        heads: int,
    ):
        super().__init__()

        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gin = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.BatchNorm1d(2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            )

            self.convs.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=gin,
                    heads=heads,
                    dropout=attn_dropout,
                    attn_type="multihead",
                )
            )

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data: torch_geometric.data.Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.cat([x, data.pe], dim=-1)
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index, batch)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)

        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, self.dropout, training=self.training)

        return self.classifier(x).view(-1)
