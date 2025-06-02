import torch
from torch import nn
import torch.nn.functional as F
import math


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


class ECALayer(nn.Module):
    """通道注意力模块（ECA-Net改进版）"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2))
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = self.proj(x)
        return self.layer_norm(x)


class ECAResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0):
        super(ECAResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = ECALayer(out_channels)  # 添加通道注意力

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)  # 应用通道注意力
        out += self.shortcut(x)
        return self.dropout(self.elu(out))


class Conv1dWithLengths(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        :param x: tuple
        :return: tuple
        """
        x, lengths = x
        x = self.conv(x)
        # 更新有效长度（假设为 valid 卷积）
        new_lengths = (lengths + 2 * self.padding - self.kernel_size) // self.stride + 1
        return x, new_lengths


class AvgPool1dWithLengths(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        :param x: tuple
        :return: tuple
        """
        x, lengths = x
        x = self.pool(x)
        # 更新有效长度（假设为 valid 卷积）
        new_lengths = (lengths + 2 * self.padding - self.kernel_size) // self.stride + 1
        return x, new_lengths


class ECAResidualBlock1dWithLengths(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ECAResidualBlock1dWithLengths, self).__init__()
        self.conv1 = Conv1dWithLengths(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = Conv1dWithLengths(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = ECALayer(out_channels)  # 添加通道注意力

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out, valid_lens = self.conv1(x)
        out = self.elu(self.bn1(out))
        out, valid_lens = self.conv2((out, valid_lens))
        out = self.bn2(out)
        out = self.eca(out)  # 应用通道注意力
        out += self.shortcut(x[0])
        return self.elu(out), valid_lens


def generate_mask(valid_lens):
    # 计算当前批次的最大有效长度（转换为整数）
    max_len = valid_lens.max().item()
    # 生成位置索引矩阵，形状为 (batch_size, max_len)
    indices = torch.arange(max_len, device=valid_lens.device).unsqueeze(0) \
        .expand(valid_lens.size(0), -1)
    # 比较每个位置是否小于对应的有效长度，生成布尔掩码并转换为浮点型
    mask = indices < valid_lens.unsqueeze(1)
    return mask


class SelfAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.SELU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, x, valid_lens):
        """
        :param x: (batch_size, seq_len, embed_dim)
        :param valid_lens: (batch_size,)
        :return: x
        """
        mask = generate_mask(valid_lens)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=~mask)
        x = self.ln(attn_out + x)
        x = self.ffn(x) + x
        return self.ln(x)


class AttnResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4, dropout=0):
        super().__init__()
        self.eca_residual_block = ECAResidualBlock1dWithLengths(in_channels, out_channels, stride)
        self.attn_block = SelfAttnBlock(embed_dim=out_channels, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, valid_lens = self.eca_residual_block(x)
        x_attn = x.permute(0, 2, 1)
        x_attn = self.attn_block(x_attn, valid_lens)
        x_attn = x_attn.permute(0, 2, 1)
        return self.dropout(x_attn + x), valid_lens


def make_layer(block_class, in_channels, out_channels, blocks, stride, dropout=0):
    layers = [block_class(in_channels, out_channels, stride=stride, dropout=dropout)]
    for _ in range(1, blocks):
        layers.append(block_class(out_channels, out_channels, stride=1, dropout=dropout))
    return nn.Sequential(*layers)


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        query = self.query.repeat(x.size(1), 1).unsqueeze(0)  # (1, batch, embed_dim)
        attn_out, _ = self.attn(query, x, x)
        return attn_out.squeeze(0)  # (batch, embed_dim)


class GatedStatisticalPool(nn.Module):
    """双向门控统计池化（Bi-directional Gated Statistical Pooling）"""
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        stats = torch.cat([mean, std], dim=-1)
        gate = self.gate(stats)
        return gate * mean + (1 - gate) * std


class STPyramidPool(nn.Module):
    """时空金字塔池化（Spatial-Temporal Pyramid Pooling）"""
    def __init__(self, embed_dim, levels=None):
        super().__init__()
        if levels is None:
            levels = [1, 3, 5, 7]
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim // len(levels), k, padding=k // 2)
            for k in levels
        ])

    def forward(self, x):
        pooled = [F.adaptive_max_pool1d(conv(x), 1) for conv in self.conv_layers]
        return torch.cat(pooled, dim=1).squeeze(-1)


class Aggregator(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()
        self.attention_pool = MultiHeadAttentionPool(embed_dim, num_heads=4)
        self.stat_pool = GatedStatisticalPool(embed_dim)
        self.st_pool = STPyramidPool(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.attention_pool(x.permute(2, 0, 1))
        h2 = self.stat_pool(x)
        h3 = self.st_pool(x)

        fused = self.dropout(torch.cat([h1, h2, h3], dim=-1))
        return self.fusion(fused)


class DynamicMasker(nn.Module):
    """动态掩码生成器"""
    def __init__(self):
        super().__init__()
        self.valid_lens = None

    def init_valid_lens(self, valid_lens):
        """
        :param valid_lens: (batch_size,)
        """
        self.valid_lens = valid_lens

    def forward(self, kernel_size, stride, padding):
        self.valid_lens = (self.valid_lens + 2 * padding - kernel_size) // stride + 1
        return self.generate_mask()

    def generate_mask(self):
        # 计算当前批次的最大有效长度（转换为整数）
        max_len = self.valid_lens.max().item()
        # 生成位置索引矩阵，形状为 (batch_size, max_len)
        indices = torch.arange(max_len, device=self.valid_lens.device).unsqueeze(0) \
            .expand(self.valid_lens.size(0), -1)
        # 比较每个位置是否小于对应的有效长度，生成布尔掩码并转换为浮点型
        mask = indices < self.valid_lens.unsqueeze(1)
        return mask


