import json
import sys
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

from common.data_loaders.data_loaders import IDSDataLoader
from common.data_loaders.encoder import ColumnEncoder


class JSONLDataLoader(IDSDataLoader):
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
        x_rows = []
        y_rows = []

        feature_order: list[str] | None = None
        target_column: str | None = None

        encoder = ColumnEncoder()

        for line_number, line in self._iter_lines(data_arg):
            line = line.strip()

            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                ) from e

            row = doc.get("contained", doc)

            if not isinstance(row, dict):
                continue

            if feature_order is None:
                target_column = self._find_target_column(row)

                feature_order = [
                    key
                    for key in row.keys()
                    if key not in self.DROP_COLUMNS
                       and key != target_column
                ]

            x_rows.append([
                encoder.encode(key, row.get(key))
                for key in feature_order
            ])

            if target_column is not None:
                y_rows.append(
                    self._target_to_int(row.get(target_column))
                )

        if not x_rows:
            raise ValueError(f"No valid records found in JSONL data: {data_arg}")

        x_tensor = torch.tensor(x_rows, dtype=torch.float32)

        if target_column is not None:
            y_tensor = torch.tensor(y_rows, dtype=torch.long)
            dataset = TensorDataset(x_tensor, y_tensor)
        else:
            dataset = TensorDataset(x_tensor)

        dataset.columns = feature_order
        dataset.encoders = encoder.category_maps

        return DataLoader(dataset)

    def _iter_lines(self, data_arg: str) -> Iterable[tuple[int, str]]:
        if data_arg == "-":
            for line_number, line in enumerate(sys.stdin, start=1):
                yield line_number, line
            return

        with open(data_arg, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                yield line_number, line

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
