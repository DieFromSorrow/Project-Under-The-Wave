import torch
from torch.nn.utils.rnn import pad_sequence


def wave_pad_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    inputs = inputs.unsqueeze(dim=1)
    labels = torch.tensor(labels)
    return inputs, labels


def mfcc_pad_collate_fn(batch):
    data_batch, label_batch = [], []

    # 解包数据并记录原始长度
    for data, label in batch:
        squeezed_data = data.squeeze(0)  # 移除可能的单维度
        data_batch.append(squeezed_data)
        label_batch.append(label)

    # 自动补齐时间维度
    padded_data = pad_sequence(data_batch, batch_first=True, padding_value=0)

    # 调整维度顺序 -> [batch, channels, time]
    padded_data = padded_data.transpose(1, 2)

    # 转换为Tensor
    label_tensor = torch.tensor(label_batch)
    return padded_data, label_tensor


def mfcc_pad_collate_with_lens_fn(batch):
    data_batch, label_batch, lengths = [], [], []

    # 解包数据并记录原始长度
    for data, label in batch:
        squeezed_data = data.squeeze(0)  # 移除可能的单维度
        data_batch.append(squeezed_data)
        label_batch.append(label)
        lengths.append(squeezed_data.size(0))  # 记录原始时间步长

    # 自动补齐时间维度
    padded_data = pad_sequence(data_batch, batch_first=True, padding_value=0)

    # 调整维度顺序 -> [batch, channels, time]
    padded_data = padded_data.transpose(1, 2)

    # 转换为Tensor
    label_tensor = torch.tensor(label_batch)

    valid_lens = torch.tensor(lengths)
    return (padded_data, valid_lens), label_tensor


def get_pad_collate_fn(need_valid_lens):
    if need_valid_lens:
        return mfcc_pad_collate_with_lens_fn
    else:
        return mfcc_pad_collate_fn
