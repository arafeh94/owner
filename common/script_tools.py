import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from common.data_loaders.data_loaders import DatasetsDataLoader, IDSDataLoader
from common.data_loaders.inline_json_data_loader import InlineJSONDataLoader
from common.data_loaders.jsonl_data_loader import JSONLDataLoader
from common.data_loaders.mongo_data_loader import StreamScanMongoDBDataLoader
from common.ml_args import MLArgs
from module import EXTRACTOR_PATH


def try_parse_json(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extractor() -> list[str]:
    if not EXTRACTOR_PATH.exists():
        return []

    with open(EXTRACTOR_PATH, "r", encoding="utf-8") as f:
        value = json.load(f)

    if not value:
        return []

    if not isinstance(value, list):
        raise ValueError("extractor.json must be a JSON array")

    return value


# noinspection PyUnresolvedReferences
def apply_extractor(data: DataLoader, extractor) -> DataLoader:
    if not extractor:
        return data

    dataset = data.dataset

    if not isinstance(dataset, TensorDataset):
        raise TypeError(
            "Extractor cannot be applied: unsupported dataset type. "
            "Only DataLoader instances backed by TensorDataset are supported."
        )

    if not hasattr(dataset, "columns"):
        raise ValueError(
            "Extractor cannot be applied: this dataset does not support feature "
            "extraction because it has no `columns` metadata."
        )

    columns = dataset.columns

    indices = []
    selected_columns = []
    missing_fields = []

    for field in extractor:
        # Case 1: normal field name, e.g. "src_port"
        if isinstance(field, str):
            candidates = [field]

        # Case 2: aliases, e.g. ["src_ip", "source_ip"]
        elif isinstance(field, list) and all(isinstance(item, str) for item in field):
            candidates = field

        else:
            raise ValueError(
                "Invalid extractor field. Each item must be either a string "
                f"or a list of strings. Got: {field!r}"
            )

        matched_column = next(
            (candidate for candidate in candidates if candidate in columns),
            None,
        )

        if matched_column is None:
            missing_fields.append(candidates)
            continue

        indices.append(columns.index(matched_column))
        selected_columns.append(matched_column)

    if not indices:
        raise ValueError(
            "Extractor did not match any dataset columns. "
            f"Requested: {extractor}. Available: {columns}."
        )

    tensors = dataset.tensors

    if len(tensors) == 1:
        x = tensors[0]
        y = None
    elif len(tensors) == 2:
        x, y = tensors
    else:
        raise ValueError(
            f"Unsupported TensorDataset format. Expected 1 or 2 tensors, got {len(tensors)}"
        )

    x_selected = x[:, indices]

    if y is None:
        y = torch.full(
            size=(x_selected.shape[0],),
            fill_value=1,
            dtype=torch.long,
        )

    new_dataset = TensorDataset(x_selected, y)
    new_dataset.columns = selected_columns

    return DataLoader(
        new_dataset,
        batch_size=data.batch_size,
        shuffle=False,
    )


def get_data_loader(data_type: str) -> IDSDataLoader:
    data_loader = None
    if data_type == "dataset":
        data_loader = DatasetsDataLoader()

    if data_type == "mongo":
        data_loader = StreamScanMongoDBDataLoader()

    if data_type == "jsonl":
        data_loader = JSONLDataLoader()

    if data_type == "inline-json":
        data_loader = InlineJSONDataLoader()
    return data_loader


def resolve_data(data_arg: str, data_type: str) -> DataLoader:
    dl = get_data_loader(data_type=data_type)
    if not dl:
        raise Exception(f"Invalid data type: {data_type}")
    return dl.load(data_arg)


def prepare_data(args: MLArgs) -> DataLoader:
    data = resolve_data(args.data, args.data_type)
    extractor = load_extractor()
    return apply_extractor(data, extractor)


def run_script(
        root: Path,
        script_name: str,
        ml_args: MLArgs,
        console_display: bool = True,
):
    stdin_data = None

    if ml_args.data == "-":
        stdin_data = sys.stdin.read()

    process = subprocess.Popen(
        [
            sys.executable,
            str(root / script_name),
            "--run", ml_args.run,
            "--data", ml_args.data,
            "--data-type", ml_args.data_type,
            "--args", json.dumps(ml_args.args),
        ],
        cwd=root,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if stdin_data is not None and process.stdin is not None:
        process.stdin.write(stdin_data)
        process.stdin.close()

    output_lines = []
    last_line = ""

    if process.stdout is not None:
        for line in process.stdout:
            output_lines.append(line)

            if console_display:
                print(line, end="")

            if line.strip():
                last_line = line.strip()

    process.wait()

    stdout_text = "".join(output_lines)

    if process.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code {process.returncode}\n\n"
            f"Command:\n{' '.join(map(str, process.args))}\n\n"
            f"OUTPUT:\n{stdout_text}"
        )

    return last_line
