from common.ml_args import MLArgs
from common.script_tools import prepare_data

ml_args = MLArgs({"run": "detect", "data": "atlas", "data_type": "mongo"})

data = prepare_data(ml_args)
print(data)
