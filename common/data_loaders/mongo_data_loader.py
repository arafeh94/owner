import os
from typing import Any

import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import DataLoader, TensorDataset

from common.data_loaders.data_loaders import IDSDataLoader
from common.data_loaders.encoder import ColumnEncoder

load_dotenv()


class StreamScanMongoDBDataLoader(IDSDataLoader):
    DROP_COLUMNS = {
        "id",
        "_id",
        "model_name",
    }

    TARGET_COLUMNS = {
        "Label",
        "label",
        "Attack",
        "attack",
        "target",
        "Target",
        "y",
    }

    def __init__(
            self,
            mongo_uri: str | None = None,
            database_name: str | None = None,
    ):
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017",
        )

        self.database_name = database_name or os.getenv(
            "MONGODB_DATABASE",
            "ml_s",
        )

    def load(self, data_arg: str) -> DataLoader:
        """
        data_arg = MongoDB collection name.

        Example:
            --data atlas
            --data-type mongo
        """

        client = MongoClient(self.mongo_uri)

        try:
            collection = client[self.database_name][data_arg]

            docs = list(collection.find({}, {"_id": 0}))

            if not docs:
                raise ValueError(
                    f"No documents found in MongoDB collection '{data_arg}'"
                )

            rows = []

            for doc in docs:
                row = doc.get("contained", doc)

                if isinstance(row, dict):
                    rows.append(row)

            if not rows:
                raise ValueError(
                    f"No valid documents found in MongoDB collection '{data_arg}'"
                )

            target_column = self._find_target_column(rows[0])

            feature_order = [
                key
                for key in rows[0].keys()
                if key not in self.DROP_COLUMNS
                and key != target_column
            ]

            encoder = ColumnEncoder()

            x_rows = []
            y_rows = []

            for row in rows:
                x_rows.append([
                    encoder.encode(key, row.get(key))
                    for key in feature_order
                ])

                if target_column is not None:
                    y_rows.append(
                        self._target_to_int(row.get(target_column))
                    )

            x_tensor = torch.tensor(x_rows, dtype=torch.float32)

            if target_column is not None:
                y_tensor = torch.tensor(y_rows, dtype=torch.long)
            else:
                y_tensor = torch.full(
                    size=(x_tensor.shape[0],),
                    fill_value=-1,
                    dtype=torch.long,
                )

            dataset = TensorDataset(x_tensor, y_tensor)

            dataset.columns = feature_order
            dataset.encoders = encoder.category_maps

            return DataLoader(dataset)

        finally:
            client.close()

    def _find_target_column(self, row: dict[str, Any]) -> str | None:
        for key in row.keys():
            if key in self.TARGET_COLUMNS:
                return key

        return None

    def _target_to_int(self, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        text = str(value).strip()

        if text.isdigit():
            return int(text)

        normalized = text.lower()

        if normalized in {"benign", "normal", "false", "no"}:
            return 0

        if normalized in {"attack", "malicious", "true", "yes"}:
            return 1

        raise ValueError(
            f"Unsupported target value {value!r}. "
            "Use numeric labels or handle label mapping in preprocessing."
        )