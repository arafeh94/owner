import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common import dataloader_tools
from common.ml_args import MLArgs
from common.ml_results import train_ok
from common.script_tools import prepare_data
from module import ROOT
from src import model_ids, model_mnist

ml_args = MLArgs({'run': 'train', 'data': 'ids', 'data-type': 'dataset', 'args': {'epochs': 2, 'lr': 1e-3}})

data: DataLoader = prepare_data(ml_args)
extra_args = ml_args.args


# Only modify this functions
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = dataloader_tools.split_loader(data)

    sample_inputs, _ = next(iter(train_data))
    input_dim = sample_inputs.shape[1]

    model = model_ids.IDSNet(input_dim=input_dim, num_classes=2).to(device)

    lr = extra_args.get("lr", 1e-3)
    epochs = extra_args.get("epochs", 5)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = model_ids.train(model, train_data, optimizer, criterion, device)
        test_loss, test_acc = model_ids.evaluate(model, test_data, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

    model_path = ROOT / "model.pt"
    torch.save(model.state_dict(), model_path)
    return train_ok(model_path)


# old ignore
def bkp_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_mnist.Net().to(device)
    train_data, test_data = dataloader_tools.split_loader(data)
    lr = extra_args.get("lr", 1e-3)
    epochs = extra_args.get("epochs", 5)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = model_mnist.train(model, train_data, optimizer, criterion, device)
        test_loss, test_acc = model_mnist.evaluate(model, test_data, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

    model_path = ROOT / "model.pt"
    torch.save(model.state_dict(), model_path)
    return train_ok(model_path)


if __name__ == "__main__":
    print(json.dumps(main()))
