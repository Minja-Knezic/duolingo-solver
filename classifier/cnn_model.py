import torch


# --- Simple CNN ---
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)  # grayscale input
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # -> [batch, 16, 32, 32]
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  # -> [batch, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

