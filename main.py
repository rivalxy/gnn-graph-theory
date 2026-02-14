import argparse
import torch
import torch.nn as nn
import torch_geometric
import pandas as pd

from torch_geometric.loader import DataLoader
from sklearn import metrics
from model import GIN


def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    predictions = []
    labels = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = (out > 0).float()
        predictions.append(pred.cpu())
        labels.append(data.y.cpu())

    accuracy = metrics.accuracy_score(torch.cat(labels), torch.cat(predictions))
    f1 = metrics.f1_score(torch.cat(labels), torch.cat(predictions))

    return accuracy, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIN for partial automorphism extension problem")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)") 
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Input batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0027324691985734575,
                        help="Learning rate (default: 0.0027324691985734575)")
    parser.add_argument("--weight_decay", type=float, default=3.47e-06,
                        help="Weight decay (default: 3.5e-06)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension size (default: 512)") 
    parser.add_argument("--num_layers", type=int, default=3, 
                        help="Number of GIN layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.6, 
                        help="Dropout rate (default: 0.6)")
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
    model = GIN(3, args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.factor, patience=args.patience
    )

    # train_loss, train_acc, train_f1, val_acc, val_f1
    best_model_stats = [0.0, 0.0, 0.0, 0.0, 0.0]
    training_history = []
    patience = 15
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        train_acc, train_f1 = test(train_loader)
        val_acc, val_f1 = test(val_loader)
        scheduler.step(val_acc)
        
        training_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "learning_rate": optimizer.param_groups[0]['lr']   
        })

        if val_acc > best_model_stats[3]:
            best_model_stats = [train_loss,
                                train_acc, train_f1, val_acc, val_f1]
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Train F1:  {train_f1:.4f} | "
              f"Val Acc:   {val_acc:.4f} | "
              f"Val F1:    {val_f1:.4f}")


    history_df = pd.DataFrame(training_history)
    history_df.to_csv("results/training_history.csv", index=False)

    print("================================\n")
    print("Best Model Stats:")
    print(f"Train Loss: {best_model_stats[0]:.4f} | "
          f"Train Acc: {best_model_stats[1]:.4f} | "
          f"Train F1:  {best_model_stats[2]:.4f} | "
          f"Val Acc:   {best_model_stats[3]:.4f} | "
          f"Val F1:    {best_model_stats[4]:.4f}")
