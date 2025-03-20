import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random


class FrequencyMask:
    def __init__(self, max_width=5):
        self.max_width = max_width

    def __call__(self, spec):
        n_mels, time = spec.shape
        width = random.randint(1, self.max_width)
        start = random.randint(0, n_mels - width)
        spec[start:start+width, :] = 0
        return spec


class TimeMask:
    def __init__(self, max_width=128):
        self.max_width = max_width

    def __call__(self, spec):
        time, n_mels = spec.shape
        width = random.randint(1, self.max_width)
        start = random.randint(0, time - width)
        spec[:, start:start+width] = 0
        return spec


class TimeStretch:
    def __init__(self, max_rate=0.05):
        self.max_rate = max_rate

    def __call__(self, spec):
        rate = 1.0 + random.uniform(-self.max_rate, self.max_rate)
        n_mels, orig_time = spec.shape
        new_time = int(orig_time * rate)
        spec = spec.unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,time)
        spec = F.interpolate(spec, size=(n_mels, new_time), mode='bilinear')
        spec = spec.squeeze()
        if new_time > orig_time:
            spec = spec[:, :orig_time]
        else:
            spec = F.pad(spec, (0, orig_time - new_time))
        return spec


class PitchShift:
    def __init__(self, max_shift=2):
        self.max_shift = max_shift

    def __call__(self, spec):
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        return torch.roll(spec, shifts=shift, dims=0)


class TimeShift:
    def __init__(self, max_shift=100):
        self.max_shift = max_shift

    def __call__(self, spec):
        _, time = spec.shape
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        return torch.roll(spec, shifts=shift, dims=1)


class AddNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, spec):
        noise = torch.randn_like(spec) * self.noise_level
        return spec + noise


class RandomGain:
    def __init__(self, min_gain=0.8, max_gain=1.2):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, spec):
        gain = random.uniform(self.min_gain, self.max_gain)
        return spec * gain


# 组合增强
class Augmentor:
    def __init__(self):
        self.transforms = [
            FrequencyMask(),
            TimeMask(),
            TimeStretch(),
            PitchShift(),
            TimeShift(),
            AddNoise(),
            RandomGain()
        ]

    def __call__(self, spec):
        for t in self.transforms:
            if random.random() < 0.5:  # 50%概率应用每个增强
                spec = t(spec)
        return spec


# 数据集示例
class MelDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spec = self.data[idx]  # 假设输入为(n_mels, time)
        if self.transform:
            spec = self.transform(spec)
        return spec


# 使用示例
if __name__ == "__main__":
    dummy_data = [torch.randn(128, 500) for _ in range(10)]  # 模拟10个样本
    dataset = MelDataset(dummy_data, transform=Augmentor())
    sample = dataset[0]
    print("Augmented Spec Shape:", sample.shape)
