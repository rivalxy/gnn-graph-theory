import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE

PE_DIM = 5

train_dataset = torch.load("dataset/train_dataset.pt", weights_only=False)
val_dataset = torch.load("dataset/val_dataset.pt", weights_only=False)
test_dataset = torch.load("dataset/test_dataset.pt", weights_only=False)

transform = AddLaplacianEigenvectorPE(k=PE_DIM, attr_name="laplacian_eigenvector_pe")

train_dataset = [transform(data) for data in train_dataset]
val_dataset = [transform(data) for data in val_dataset]
test_dataset = [transform(data) for data in test_dataset]

torch.save(train_dataset, "dataset/train_dataset_with_pe.pt")
torch.save(val_dataset, "dataset/val_dataset_with_pe.pt")
torch.save(test_dataset, "dataset/test_dataset_with_pe.pt")
