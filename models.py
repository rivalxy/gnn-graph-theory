import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_heads, pe_dim):
        """
        GPSConv model with LaplacianPE.

        pe_dim: number of Laplacian eigenvectors (fixed at PE_DIM=5).
        """
        super().__init__()
        self.dropout = dropout
        self.pe_dim = pe_dim

        # Projects the first pe_dim eigenvectors into hidden_dim
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)

        # Brings raw node features up to hidden_dim before GPS layers
        self.input_encoder = nn.Linear(input_dim, hidden_dim)

        # GPS layers: each wraps a local GINConv + global MultiheadAttention
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=GINConv(gin_mlp),
                    heads=num_heads,
                    dropout=dropout,
                    attn_type="multihead",
                )
            )

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        pe = data.laplacian_eigenvector_pe
        x = self.input_encoder(x) + self.pe_encoder(pe)

        # GPS layers with jumping-knowledge sum over layers
        h_graph = 0
        for conv in self.convs:
            x = conv(x, edge_index, batch)
            x = F.dropout(x, self.dropout, training=self.training)
            h_graph = h_graph + global_add_pool(x, batch)

        h = F.relu(self.lin1(h_graph))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.classifier(h).view(-1)