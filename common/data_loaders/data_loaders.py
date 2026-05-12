from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from common import datasets as datasets_module


class IDSDataLoader(ABC):
    @abstractmethod
    def load(self, data_arg: str) -> DataLoader:
        ...


class DatasetsDataLoader(IDSDataLoader):
    def load(self, data_arg: str):
        if not hasattr(datasets_module, data_arg):
            raise ValueError(f"Dataset '{data_arg}' not found in datasets.py")

        dataset_fn = getattr(datasets_module, data_arg)

        if not callable(dataset_fn):
            raise ValueError(f"'{data_arg}' exists but is not callable")

        return dataset_fn()

    def __init__(self):
        pass

