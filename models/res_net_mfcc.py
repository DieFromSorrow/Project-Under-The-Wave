from torch import nn
from models.blocks import ResidualBlock1d
from models.blocks import MyGRU
from models.blocks import build_layer


class ResNetMfccClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, layers, gru_hidden_size):
        super(ResNetMfccClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )

        self.res_layer1 = build_layer(
            ResidualBlock1d, in_channels=64, out_channels=64, num_blocks=layers[0], stride=1
        )
        self.res_layer2 = build_layer(
            ResidualBlock1d, in_channels=64, out_channels=128, num_blocks=layers[1], stride=2
        )
        self.res_layer3 = build_layer(
            ResidualBlock1d, in_channels=128, out_channels=256, num_blocks=layers[2], stride=2
        )
        self.res_layer4 = build_layer(
            ResidualBlock1d, in_channels=256, out_channels=512, num_blocks=layers[3], stride=2
        )

        self.gru = MyGRU(input_size=512, hidden_size=gru_hidden_size, num_layers=2, batch_first=True)

        self.fc = nn.Linear(in_features=gru_hidden_size, out_features=num_classes)
        pass

    def forward(self, x):
        output = x
        output = self.conv1(output)
        output = self.res_layer1(output)
        output = self.res_layer2(output)
        output = self.res_layer3(output)
        output = self.res_layer4(output)
        output = self.gru(output)
        output = self.fc(output)
        return output


class ResNetMfccClassifier4Test(ResNetMfccClassifier):
    def __init__(self, *args, **kwargs):
        super(ResNetMfccClassifier4Test, self).__init__(*args, **kwargs)
        self.output_list = []
        pass

    def forward(self, x):
        output = self.conv1(x)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.res_layer1(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.res_layer2(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.res_layer3(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.res_layer4(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.gru(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        output = self.fc(output)
        print(output.shape)
        self.output_list.append(output.clone().detach())

        return output, self.output_list


def resnet18_mfcc_classifier(in_channels, num_classes):
    return ResNetMfccClassifier(in_channels, num_classes, layers=[2, 2, 2, 2], gru_hidden_size=1024)


def resnet34_mfcc_classifier(in_channels, num_classes):
    return ResNetMfccClassifier(in_channels, num_classes, layers=[3, 4, 6, 3], gru_hidden_size=1024)


def resnet18_mfcc_classifier4test(in_channels, num_classes):
    return ResNetMfccClassifier4Test(in_channels, num_classes, layers=[2, 2, 2, 2], gru_hidden_size=1024)


def resnet34_mfcc_classifier4test(in_channels, num_classes):
    return ResNetMfccClassifier4Test(in_channels, num_classes, layers=[3, 4, 6, 3], gru_hidden_size=1024)
