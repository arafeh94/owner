import os

import numpy as np
import pandas as pd
import torch
from pymongo import MongoClient
from torch.utils.data import DataLoader, TensorDataset

from common.data_loaders.data_loaders import IDSDataLoader


class StreamScanMongoDBDataLoader(IDSDataLoader):
    def __init__(
        self,
        mongo_uri: str | None = None,
        database_name: str | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI",
            "mongodb://192.168.186.69:27017",
        )

        self.database_name = database_name or os.getenv(
            "MONGODB_DATABASE",
            "ml_s",
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self, data_arg: str) -> DataLoader:
        """
        data_arg is the MongoDB collection name.

        Example:
            --data atlas
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
                if "contained" not in doc:
                    continue

                rows.append(doc["contained"])

            if not rows:
                raise ValueError(
                    f"No documents with `contained` found in collection '{data_arg}'"
                )

            df = pd.DataFrame(rows)

            if "id" in df.columns:
                df.drop(columns="id", inplace=True)

            df.dropna(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            if "Label" not in df.columns:
                raise ValueError("MongoDB data must contain `Label` column")

            df["Attack"] = np.where(df["Label"] == "BENIGN", 0, 1)

            y = df["Attack"].astype(np.int64)

            drop_columns = ["Label", "Attack"]

            if "model_name" in df.columns:
                drop_columns.append("model_name")

            x = df.drop(drop_columns, axis=1)

            x = x.astype(np.float32)

            x_tensor = torch.tensor(x.values, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.long)

            dataset = TensorDataset(x_tensor, y_tensor)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )

        finally:
            client.close()