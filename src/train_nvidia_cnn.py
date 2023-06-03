import torch


def train(net, trainloader, valloader, optimizer, criterion, epochs, device="cuda"):
    net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            imgs, forward_signal, left_signal = data[0].to(device), data[1].to(device), data[2].to(device)

            optimizer.zero_grad()
            signals = torch.cat((forward_signal.unsqueeze(0), left_signal.unsqueeze(0)), 0)
            signals = torch.transpose(signals, 0, 1)
            outputs = net(imgs)
            loss = criterion(outputs, signals)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
