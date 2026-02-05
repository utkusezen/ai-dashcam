from torch import nn

class TrafficSignCNN(nn.Module):
    """
    A Neural Network class that uses convolutional feature extractors and fully connected classifier layers
    for traffic sign recognition
    """
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.feature_extractor_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor_layers(x)
        x = self.classifier_layers(x)
        return x