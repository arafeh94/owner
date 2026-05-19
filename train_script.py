import json

from torch.utils.data import DataLoader

from common.ml_args import MLArgs
from common.script_tools import prepare_data
from src import train

ml_args = MLArgs({'run': 'train', 'data': 'ids_sample', 'data-type': 'dataset', 'args': {'epochs': 2, 'lr': 1e-3}})

data: DataLoader = prepare_data(ml_args)
extra_args = ml_args.args

if __name__ == "__main__":
    print(json.dumps(train.main(data, **extra_args)))
