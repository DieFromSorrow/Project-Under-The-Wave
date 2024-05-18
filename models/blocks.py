import torch
from torch import nn


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1d, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.identity = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.identity = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

            # self.identity = nn.Sequential(
            #     nn.AvgPool1d(kernel_size=3, stride=stride, padding=1),
            #     nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            #     nn.BatchNorm1d(out_channels)
            # )

    def forward(self, x):
        identity = self.identity(x)
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += identity
        output = self.relu(output)
        return output


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(MyGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor):
        if self.batch_first:
            x = x.transpose(1, 2)
        else:
            x = x.permute(2, 0, 1)
        _, h_n = self.gru(x)
        return h_n[-1, :, :]


def build_layer(block, in_channels, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
    return nn.Sequential(*layers)
