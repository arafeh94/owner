from torch.utils.data import DataLoader, random_split


def split_loader(loader: DataLoader, train_ratio=0.8):
    dataset = loader.dataset

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
