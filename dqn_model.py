#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc_mean = nn.Linear(512, 1)
        self.fc_adv = nn.Linear(512, num_actions)

        # Initialize the parameters
        self._initialize_weights()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        # adapt to env
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x = x.permute((0, 3, 1, 2))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))

        v = self.fc_mean(x)
        adv = self.fc_adv(x)
        adv_norm = adv - adv.mean()

        return adv_norm + v

    def _initialize_weights(self):
        # Initialize convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)

        # Initialize fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.constant_(self.fc_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_adv.weight)
        nn.init.constant_(self.fc_adv.bias, 0)
