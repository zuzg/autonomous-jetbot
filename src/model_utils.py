import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.onnx


def export_onnx(model: nn.Module, path: str, device: str) -> None:
    x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
    torch.onnx.export(model,
                    x,
                    path,
                    export_params=True,
                    opset_version=11)
    

def plot_loss(epochs: int, train_history: list, val_history: list) -> None:
    plt.plot(np.arange(epochs), train_history, marker="o")
    plt.plot(np.arange(epochs), val_history, marker="o")
    plt.legend(["train", "val"])
    plt.show()
