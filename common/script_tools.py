import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, TensorDataset

from common import datasets as datasets_module
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
def apply_extractor(data: Any, extractor: list[str]) -> DataLoader:
    if not isinstance(data, DataLoader):
        raise TypeError("apply_extractor only supports DataLoader")

    if not extractor:
        return data

    dataset = data.dataset

    if not isinstance(dataset, TensorDataset):
        raise TypeError(
            "Extractor cannot be applied: unsupported dataset type. "
            "Only DataLoader instances backed by TensorDataset are supported. "
            "Ensure your dataset is converted to a TensorDataset or use a compatible dataset."
        )

    if not hasattr(dataset, "columns"):
        raise ValueError(
            "Extractor cannot be applied: this dataset does not support feature extraction "
            "because it has no `columns` metadata. "
            "A compatible dataset must define `dataset.columns = [...]`. "
            "Either update `extractor.json` to an empty array or use a supported dataset."
        )

    x, y = dataset.tensors
    columns = list(dataset.columns)

    selected_indices = [
        i for i, column in enumerate(columns)
        if column in extractor
    ]

    if not selected_indices:
        raise ValueError(
            "Extractor did not match any dataset columns. "
            "Ensure that the fields in `extractor.json` exist and match the dataset feature names. "
            f"Requested: {extractor}. "
            f"Available: {list(columns)}."
        )

    new_x = x[:, selected_indices]
    new_dataset = TensorDataset(new_x, y)
    new_dataset.columns = [columns[i] for i in selected_indices]

    return DataLoader(
        new_dataset,
        batch_size=data.batch_size,
        shuffle=True,
    )


def resolve_data(data_arg: str, data_type: str) -> DataLoader:
    if data_type != "dataset":
        raise ValueError("Currently only data_type='dataset' is supported")

    # Check if function exists in datasets.py
    if not hasattr(datasets_module, data_arg):
        raise ValueError(f"Dataset '{data_arg}' not found in datasets.py")

    dataset_fn = getattr(datasets_module, data_arg)

    if not callable(dataset_fn):
        raise ValueError(f"'{data_arg}' exists but is not callable")

    return dataset_fn()


def prepare_data(args: MLArgs) -> DataLoader:
    data = resolve_data(args.data, args.data_type)
    extractor = load_extractor()
    return apply_extractor(data, extractor)


def run_script(root: Path, script_name: str, ml_args: MLArgs, console_display: bool = True):
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / script_name),
            "--run", ml_args.run,
            "--data", ml_args.data,
            "--data-type", ml_args.data_type,
            "--args", json.dumps(ml_args.args),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_line = ""

    for line in process.stdout:
        if console_display:
            print(line, end="")

        if line.strip():
            last_line = line.strip()

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)

    return last_line
