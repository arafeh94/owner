import ipaddress
from typing import Any


class ColumnEncoder:
    def __init__(self):
        self.category_maps: dict[str, dict[str, int]] = {}

    def encode(self, column: str, value: Any) -> float:
        if value is None:
            return 0.0

        if isinstance(value, bool):
            return float(int(value))

        if isinstance(value, int | float):
            return float(value)

        text = str(value).strip()

        if text == "":
            return 0.0

        # numeric string
        try:
            return float(text)
        except ValueError:
            pass

        # IP address string
        try:
            return float(int(ipaddress.ip_address(text)))
        except ValueError:
            pass

        # categorical string
        return float(self._encode_category(column, text))

    def _encode_category(self, column: str, value: str) -> int:
        if column not in self.category_maps:
            self.category_maps[column] = {}

        mapping = self.category_maps[column]

        if value not in mapping:
            mapping[value] = len(mapping) + 1

        return mapping[value]
