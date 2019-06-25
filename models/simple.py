

import torch.nn as nn


def flatten(x):
    return x.view(x.shape[0], -1)


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init 
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)
    
    def forward(self, x):
        # forward always defines connectivity
        x = flatten(x)
        return self.fc2(F.relu(self.fc1(x)))


class ThreeLayerConvNet(nn.Module):
    """Structure: Conv|ReLU - Conv|ReLU - FC
    """
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc3 = nn.Linear(32 * 32 * channel_2, num_classes)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x)
        return self.fc3(x)


class MNIST_SimpleCNN(nn.Module):
    """Structure: CONV1 - CONV2 - FC1 - FC2
        input: (, 1, 28, 28)
        conv1-: (, 10, 12, 12)
        conv2-: (, 20, 4, 4)
        fc1-: (, 50)
        fc2-: (, 10)
    """
    def __init__(self):
        super().__init__()
        # conv layers: feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # fc layers: classifier
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = flatten(x)
        return self.fc_layers(x)