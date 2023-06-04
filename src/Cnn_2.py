import torch.nn as nn

class CNN_2(nn.Module):
    """LeNet architecture."""

    def __init__(self):
        """Initialization."""
        super(CNN_2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 36, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(36, 48, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.4)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=25600, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output.double()