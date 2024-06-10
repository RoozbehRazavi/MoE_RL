import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LunarLanderCNN(NatureCNN):
    def __init__(self, *args, mixture, **kwargs):
        super(LunarLanderCNN, self).__init__(*args)
        channel = mixture.model.state_encoder.state_embedding.layers.channel
        self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, channel, kernel_size=3, stride=1),
        nn.BatchNorm2d(channel),
        nn.ReLU(),
        nn.Flatten()
        )

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(160, 8, 4), 4, 2), 3, 1)
        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(210, 8, 4), 4, 2), 3, 1)

    def forward(self, x):
        x = self.cnn(x)
        # TODO should I permute the input?
        return x


import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Define each layer explicitly
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 22 * 16, features_dim)  # This assumes the final feature map is 7x7
        self.relu_final = nn.ReLU()

    def forward(self, observations):
        # breakpoint()
        x = self.conv1(observations)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu_final(x)

        return x
