from dataclasses import asdict, dataclass
from typing import List


def train_ok(model_path: str, **args):
    return {
        "status": "ok",
        "model_path": model_path,
        **args
    }


def train_failed(error, **args):
    return {
        "status": "failed",
        "error": error,
        **args
    }


def predict_ok(results: 'PredictionResults'):
    return results.to_dict()

@dataclass
class Record:
    y: int
    acc: float

class PredictionResults:

    def __init__(self):
        self.results: List["Record"] = []

    def append(self, result: "Record"):
        self.results.append(result)

    def to_dict(self):
        return [
            asdict(result)
            for result in self.results
        ]
