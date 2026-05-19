import json

from common.ml_args import MLArgs
from common.script_tools import prepare_data
from src import predict

ml_args = MLArgs({"run": "detect", "data": "ids_sample", "data-type": "dataset"})

data = prepare_data(ml_args)
extra_args = ml_args.args


def main():
    return predict.main(data, **extra_args)


if __name__ == "__main__":
    print(json.dumps(main()))
