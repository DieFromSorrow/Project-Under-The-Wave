from models.blocks import make_layer, SelfAttnBlock, AttnResidualBlock, \
    Aggregator, Conv1dWithLengths, AvgPool1dWithLengths
from torch import nn


class AttnResNetMfccClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, layers, dropout=0.2):
        super().__init__()
        self.base_conv = Conv1dWithLengths(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.avg_pooling = AvgPool1dWithLengths(kernel_size=2, stride=2, padding=0)

        self.attn_res_layers = nn.Sequential(
            make_layer(AttnResidualBlock, 64, 64, layers[0], stride=1, dropout=dropout),
            make_layer(AttnResidualBlock, 64, 128, layers[1], stride=2, dropout=dropout),
            make_layer(AttnResidualBlock, 128, 256, layers[2], stride=2, dropout=dropout),
            make_layer(AttnResidualBlock, 256, 512, layers[3], stride=2, dropout=dropout)
        )

        self.aggregator = Aggregator(512, dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        :param x: A tuple of (x, valid_lens)
        :return: Score tensor (batch_size, num_classes)
        """
        x, valid_lens = self.base_conv(x)
        x = self.bn1(x)
        x = self.avg_pooling((x, valid_lens))
        x, valid_lens = self.attn_res_layers(x)
        feat = self.aggregator(x)
        return self.fc(feat)


def attn_resnet18_mfcc_classifier(in_channels, num_classes):
    return AttnResNetMfccClassifier(in_channels, num_classes, layers=[1, 1, 2, 1])
