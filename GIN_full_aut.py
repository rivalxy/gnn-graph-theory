import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric import seed_everything
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader

seed_everything(42)

train_dataset = torch.load("dataset/train_dataset.pt", weights_only=False)
val_dataset = torch.load("dataset/val_dataset.pt", weights_only=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)


class GIN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=5, dropout=0.2):
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ))
            )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.classifier(x).view(-1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GIN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)


def train_epoch():
    model.train()
    total_loss = 0
    total_samples = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / total_samples


@torch.no_grad()
def eval_epoch(loader):
    model.eval()
    total_correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = (out > 0).float()
        total_correct += (pred == data.y).sum().item()
        total += data.num_graphs

        all_preds.append(pred.cpu().view(-1))
        all_targets.append(data.y.cpu().float().view(-1))

    acc = total_correct / total
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    denom = (2 * tp + fp + fn)
    f1 = 0.0 if denom == 0 else (2 * tp) / denom

    return acc, f1


if __name__ == "__main__":
    # train_loss, train_acc, train_f1, val_acc, val_f1
    best_model_stats = [0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(1, 101):
        train_loss = train_epoch()
        train_acc, train_f1 = eval_epoch(train_loader)
        val_acc, val_f1 = eval_epoch(val_loader)
        scheduler.step(val_acc)

        if val_acc > best_model_stats[3]:
            best_model_stats = [train_loss,
                                train_acc, train_f1, val_acc, val_f1]

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train F1:  {train_f1:.4f} | "
              f"Val Acc:   {val_acc:.4f} | "
              f"Val F1:    {val_f1:.4f}")

    print("================================\n")
    print("Best Model Stats:")
    print(f"Train Loss: {best_model_stats[0]:.4f} | "
          f"Train Acc: {best_model_stats[1]:.4f} | "
          f"Train F1:  {best_model_stats[2]:.4f} | "
          f"Val Acc:   {best_model_stats[3]:.4f} | "
          f"Val F1:    {best_model_stats[4]:.4f}")
