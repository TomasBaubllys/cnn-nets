import torch
import torch.nn as nn

class Conv1dNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(124, 30),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Conv1dNetMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Conv1dNetLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 32, kernel_size=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    conv1dNet = Conv1dNetSmall()
    dummy = torch.rand(1, 1, 31)
    print(conv1dNet(dummy))