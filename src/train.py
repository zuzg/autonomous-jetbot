import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any


def reshape_signals(forward_signal: Tensor, left_signal: Tensor) -> Tensor:
    """
    Reshape signals to match label
    """
    signals = torch.cat((forward_signal.unsqueeze(0), left_signal.unsqueeze(0)), 0)
    signals = torch.transpose(signals, 0, 1)
    return signals


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: Any,
    criterion: Any,
    epoch_nr: int,
    device: str,
) -> float:
    """
    Perform training for one epoch
    """
    model.train(True)
    loop = tqdm(trainloader)
    for _, data in enumerate(loop):
        imgs, forward_signal, left_signal = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
        )

        optimizer.zero_grad()
        signals = reshape_signals(forward_signal, left_signal)
        outputs = model(imgs.float())
        loss = criterion(outputs.float(), signals.float())
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch_nr}")
        loop.set_postfix(loss=loss.item())
    return loss.item()


def eval_one_epoch(
    model: nn.Module, valloader: DataLoader, criterion: Any, device: str
) -> float:
    """
    Perform evaluation for one epoch
    """
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(valloader):
            imgs, forward_signal, left_signal = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
            )
            signals = reshape_signals(forward_signal, left_signal)
            outputs = model(imgs)
            loss = criterion(outputs, signals)
    print(f"Validation loss={loss.item()}")
    return loss.item()


def run_training(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    optimizer: Any,
    criterion: Any,
    epochs: int,
    device: str = "cuda",
) -> tuple[list, list]:
    """
    Train the model
    """
    model.to(device)
    train_history = []
    val_history = []
    print(f"Running training for {epochs} epochs.")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, trainloader, optimizer, criterion, epoch, device
        )
        val_loss = eval_one_epoch(model, valloader, criterion, device)
        train_history.append(train_loss)
        val_history.append(val_loss)

    return train_history, val_history
