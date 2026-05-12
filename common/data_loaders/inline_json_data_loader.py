import json
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from common.data_loaders.data_loaders import IDSDataLoader
from common.data_loaders.encoder import ColumnEncoder


class InlineJSONDataLoader(IDSDataLoader):
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

    def load(self, data_arg: str) -> DataLoader:
        if data_arg == "-":
            raw_text = sys.stdin.read()
        else:
            raw_text = data_arg

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid inline JSON: {e}") from e

        records = self._normalize_records(raw)

        if not records:
            raise ValueError("No records found in inline JSON")

        rows = []

        for record in records:
            row = record.get("contained", record)

            if isinstance(row, dict):
                rows.append(row)

        if not rows:
            raise ValueError("No valid JSON objects found")

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
            dataset = TensorDataset(x_tensor, y_tensor)
        else:
            dataset = TensorDataset(x_tensor)

        dataset.columns = feature_order
        dataset.encoders = encoder.category_maps

        return DataLoader(dataset)

    def _normalize_records(self, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return raw

        if isinstance(raw, dict):
            if isinstance(raw.get("records"), list):
                return raw["records"]

            if isinstance(raw.get("data"), list):
                return raw["data"]

            return [raw]

        raise ValueError("Inline JSON must be an object or array")

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
