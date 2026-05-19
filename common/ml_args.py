import argparse
import json
from typing import Any, Optional


class MLArgs:
    def __init__(self, defaults: Optional[dict[str, Any]] = None):
        defaults = defaults or {}

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--run",
            required=False,
            default=defaults.get("run"),
            choices=["train", "predict"],
        )

        parser.add_argument(
            "--data",
            required=False,
            default=defaults.get("data"),
        )

        parser.add_argument(
            "--data-type",
            required=False,
            default=defaults.get("data-type"),
            choices=["mongo", "dataset", "jsonl", "inline-json"],
        )

        parser.add_argument(
            "--args",
            default=json.dumps(defaults.get("args", {})),
        )

        parsed = parser.parse_args()

        self._require(parsed.run, "run")
        self._require(parsed.data, "data")
        self._require(parsed.data_type, "data-type")

        self.run: str = parsed.run
        self.data: str = parsed.data
        self.data_type: str = parsed.data_type
        self.args: dict[str, Any] = self._parse_json_args(parsed.args)

    def _require(self, value: Any, name: str):
        if value is None:
            raise ValueError(
                f"`{name}` must be provided either via CLI or defaults"
            )

    def _parse_json_args(self, raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"--args must be valid JSON: {e}") from e

        if not isinstance(value, dict):
            raise ValueError("--args must be a JSON object")

        return value
