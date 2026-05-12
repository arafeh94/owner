import os

import kagglehub
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


def ids() -> DataLoader:
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

    csv_path = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().startswith("unsw_nb15") and "training" in f.lower():
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

    if not csv_path:
        raise FileNotFoundError("Training CSV not found")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column")

    y = torch.tensor(df["label"].values, dtype=torch.long)

    drop_cols = ["label", "attack_cat"]
    x_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    x_df = pd.get_dummies(x_df, dummy_na=True)
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([float("inf"), float("-inf")], 0)
    x_df = x_df.fillna(0)
    x_df = x_df.astype("float32")
    x = torch.tensor(x_df.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(x, y)
    dataset.columns = list(x_df.columns)
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )


def atlas() -> DataLoader:
    return ids()


def ids_sample(n: int = 10) -> DataLoader:
    loader = ids()

    dataset = loader.dataset

    n = min(n, len(dataset))
    indices = list(range(n))  # deterministic sample

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=min(loader.batch_size, n),
        shuffle=True,
    )


def mnist() -> DataLoader:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader


def mnist_sample() -> DataLoader:
    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    images = []
    labels = []

    for i in range(10):
        x, y = dataset[i]
        images.append(x)
        labels.append(y)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    sample_dataset = TensorDataset(images, labels)

    return DataLoader(sample_dataset, batch_size=10, shuffle=False)
