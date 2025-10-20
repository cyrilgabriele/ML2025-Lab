import torch
import torch.nn as nn


class EuroSATCNN(nn.Module):
    def __init__(self, num_classes, img_height=64, img_width=64):
        super(EuroSATCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(13, 128, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 16, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 13, img_height, img_width)
            out = self.features(dummy_input)
            fc1_input_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc1_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x