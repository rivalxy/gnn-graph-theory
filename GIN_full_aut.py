import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from dataset_gen import read_graphs_from_g6, generate_partial_automorphism_graphs
from sklearn.model_selection import train_test_split

raw_graphs = read_graphs_from_g6("dataset/2000_raw_graphs.g6")

graphs_train, graphs_val = train_test_split(raw_graphs, test_size=0.2, random_state=42)

train_dataset = generate_partial_automorphism_graphs(graphs_train)
val_dataset   = generate_partial_automorphism_graphs(graphs_val)

class GIN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ))
            )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x).view(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = GIN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()


def train_epoch():
    model.train()
    total_loss = 0
    correct, total = 0, 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = (out > 0).float()
        correct += (pred == data.y).sum().item()
        total += 1
    return total_loss / total, correct / total

def eval_epoch(dataset):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data)
            pred = (out > 0).float()
            correct += (pred == data.y).sum().item()
            total += 1
    return correct / total


for epoch in range(1, 101):
    train_loss, train_acc = train_epoch()
    val_acc = eval_epoch(val_dataset)
    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc:   {val_acc:.4f}")
