import torch
import torch.nn as nn
from tqdm import tqdm


class IDSNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x.float())


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(total=len(dataloader), desc="Training", leave=False)

    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if (i + 1) % 300 == 0 or (i + 1) == len(dataloader):
            avg_loss = total_loss / (i + 1)
            acc = 100.0 * correct / total

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{acc:.2f}%"
            })

        pbar.update(1)

    pbar.close()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / len(dataloader), 100.0 * correct / total
