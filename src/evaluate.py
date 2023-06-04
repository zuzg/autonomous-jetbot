import numpy as np

import torch
import torch.nn as nn


Dataset = list[tuple[torch.Tensor, float, float]]


def evaluate(model: nn.Module, test_dataset: Dataset, device: str) -> dict[str, float]:
    """
    Evaluate model on test dataset.
    Returns dictionary with MSE for vertical, horizontal and total error,
    as well as individual prediction differences.

    :param model: model to evaluate
    :param test_dataset: dataset to evaluate on (list of tuples with image, forward signal, left signal), from `train_test_split`
    :param device: device to use for evaluation
    :return: dictionary with MSE for vertical, horizontal and total error,
    """
    model.eval()
    with torch.no_grad():
        test_X = [t[0] for t in test_dataset]
        test_y = [(t[1], t[2]) for t in test_dataset]

        test_X = torch.stack(test_X).to(device)
        test_y = torch.stack([torch.tensor(t) for t in test_y]).to(device).numpy()

        outputs = model(test_X)
        if device == "cuda":
            outputs = outputs.cpu()
        outputs = outputs.numpy()

        vertical_mse = np.mean((outputs[:, 0] - test_y[:, 0]) ** 2)
        horizontal_mse = np.mean((outputs[:, 1] - test_y[:, 1]) ** 2)
        total_mse = np.mean((outputs - test_y) ** 2)
        out_diff = outputs - test_y
    return {
        "vertical_mse": vertical_mse,
        "horizontal_mse": horizontal_mse,
        "total_mse": total_mse,
        "individual_prediction_differences": out_diff,
    }
