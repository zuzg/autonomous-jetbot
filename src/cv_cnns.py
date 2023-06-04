import torch.nn as nn


class ThreshCNN(nn.Module):
    def __init__(self):
        super(ThreshCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 112x112
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # 56x56
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),  # 20x20
            nn.BatchNorm2d(8),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 20 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.double()


class ExtraChannelCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),  # 76x76
            nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),  # 26x26
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),  # 10x10
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.double()


if __name__ == "__main__":
    print("Number of parameters in ThreshCNN: ")
    print(sum(p.numel() for p in ThreshCNN().parameters()))
    print("Number of parameters in ExtraChannelCNN: ")
    print(sum(p.numel() for p in ExtraChannelCNN().parameters()))
