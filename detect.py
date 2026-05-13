import json

import torch
import torch.nn.functional as F

from common.ml_args import MLArgs
from common.script_tools import prepare_data
from src import model_ids

ml_args = MLArgs({"run": "detect", "data": "mnist_sample", "data_type": "dataset"})

data = prepare_data(ml_args)
extra_args = ml_args.args


# Only modify this
def load_model(device):
    sample_inputs, _ = next(iter(data))
    input_dim = sample_inputs.shape[1]

    model = model_ids.IDSNet(input_dim=input_dim, num_classes=2)

    checkpoint = torch.load("model.pt", map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def detect(model, dataloader, device):
    results = []

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)

        confs, preds = probs.max(1)

        for y, acc in zip(preds, confs):
            results.append({
                "y": int(y.item()),
                "acc": float(acc.item()),
            })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    return detect(model, data, device)


if __name__ == "__main__":
    print(json.dumps(main()))
