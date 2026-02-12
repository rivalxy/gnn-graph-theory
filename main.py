import argparse
import torch
import torch.nn as nn
import torch_geometric

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
    parser = argparse.ArgumentParser(description="GIN for partial automorphism extension problem")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)") 
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Input batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (default: 1e-5)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension size (default: 128)") 
    parser.add_argument("--num_layers", type=int, default=5, 
                        help="Number of GIN layers (default: 5)")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Factor for learning rate scheduler (default: 0.5)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for learning rate scheduler (default: 3)")
    args = parser.parse_args()

    torch_geometric.seed_everything(args.seed)

    train_dataset = torch.load("dataset/train_dataset.pt", weights_only=False)
    val_dataset = torch.load("dataset/val_dataset.pt", weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GIN(args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.factor, patience=args.patience
    )

    # train_loss, train_acc, train_f1, val_acc, val_f1
    best_model_stats = [0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(1, args.epochs + 1):
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
