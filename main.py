import torch
import torch.nn as nn

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from model import GIN


def train():
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
def test(loader):
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
    seed_everything(42)

    train_dataset = torch.load("dataset/train_dataset.pt", weights_only=False)
    val_dataset = torch.load("dataset/val_dataset.pt", weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GIN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # train_loss, train_acc, train_f1, val_acc, val_f1
    best_model_stats = [0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(1, 101):
        train_loss = train()
        train_acc, train_f1 = test(train_loader)
        val_acc, val_f1 = test(val_loader)
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
