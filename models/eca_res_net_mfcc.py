import torch.nn as nn
from models.blocks import MultiHeadSelfAttention, make_layer, ECAResidualBlock1d


class ECAResNetMfccClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, layers, dropout=0.2):
        super(ECAResNetMfccClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 使用更高效的残差块结构
        self.res_layers = nn.Sequential(
            make_layer(ECAResidualBlock1d, 64, 64, layers[0], stride=1),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 64, 128, layers[1], stride=2),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 128, 256, layers[2], stride=2),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 256, 512, layers[3], stride=2),
        )

        # 时间注意力模块
        self.temporal_attn = MultiHeadSelfAttention(embed_dim=512)

        # 自适应池化替代固定尺寸输出
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)

        # 时间维度处理
        x = x.permute(0, 2, 1)  # [B, T, C]
        temporal_feat = self.temporal_attn(x)

        # 空间特征聚合
        spatial_feat = self.adaptive_pool(x.permute(0, 2, 1)).squeeze(-1)

        # 特征融合
        combined_feat = temporal_feat + spatial_feat
        return self.fc(combined_feat)


def eca_resnet18_mfcc_classifier(in_channels, num_classes):
    return ECAResNetMfccClassifier(in_channels, num_classes, layers=[2, 2, 2, 2])


def eca_resnet26_mfcc_classifier(in_channels, num_classes):
    return ECAResNetMfccClassifier(in_channels, num_classes, layers=[2, 2, 6, 2])


def eca_resnet38_mfcc_classifier(in_channels, num_classes):
    return ECAResNetMfccClassifier(in_channels, num_classes, layers=[3, 3, 9, 3])
