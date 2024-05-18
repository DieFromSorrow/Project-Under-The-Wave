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
    for data, label in batch:
        data_batch.append(data.squeeze(0))
        label_batch.append(label)

    # 使用pad_sequence对数据进行补齐
    padded_data = pad_sequence(data_batch, batch_first=True, padding_value=0).transpose(1, 2)

    # 将标签转换为tensor
    label_tensor = torch.tensor(label_batch)

    # 返回补齐后的数据和标签
    return padded_data, label_tensor

