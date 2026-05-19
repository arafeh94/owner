import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.ml_results import predict_ok, PredictionResults, Record
from src.models import model_ids


def load_model(data: DataLoader, device):
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
def main(dataloader: DataLoader, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dataloader, device)

    results = PredictionResults()

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)

        confs, preds = probs.max(1)

        for y, acc in zip(preds, confs):
            results.append(Record(int(y.item()), float(acc.item())))

    return predict_ok(results)
