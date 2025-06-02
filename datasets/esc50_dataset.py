import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class ESC50Dataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            meta_file: str = "meta/esc50.csv",
            sample_rate: int = 44100,
            n_mels: int = 128,
            n_fft: int = 2048,
            hop_length: int = 512,
            transform=None,
            target_transform=None,
            folds=None
    ):
        """
        ESC-50 数据集加载器

        参数:
            root_dir (str): 数据集根目录
            meta_file (str): 元数据文件路径（相对于root_dir）
            sample_rate (int): 目标采样率
            n_mels (int): 梅尔频谱的频带数量
            n_fft (int): FFT窗口大小
            hop_length (int): 帧移
            transform (callable): 可选的音频变换
            target_transform (callable): 可选的标签变换
            folds (list): 选择特定的fold(s)，None表示全部
        """
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, meta_file)
        self.sample_rate = sample_rate
        self.transform = transform
        self.target_transform = target_transform

        # 读取元数据
        self.metadata = pd.read_csv(self.meta_path)

        # 如果指定了fold，则过滤数据
        if folds is not None:
            if isinstance(folds, int):
                folds = [folds]
            self.metadata = self.metadata[self.metadata['fold'].isin(folds)]

        # 初始化梅尔频谱转换器
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 获取元数据
        row = self.metadata.iloc[idx]
        filename = row['filename']
        fold = row['fold']
        target = row['target']
        category = row['category']

        # 加载音频文件
        audio_path = os.path.join(self.root_dir, "audio", filename)
        waveform, sample_rate = torchaudio.load(audio_path)

        # 如果需要，重采样
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # 转换为单声道（如果有多个声道）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 转换为梅尔频谱
        mel_spec = self.mel_spectrogram(waveform)

        # 应用变换（如果有）
        if self.transform:
            mel_spec = self.transform(mel_spec)

        # 处理标签
        label = torch.tensor(target, dtype=torch.long)
        if self.target_transform:
            label = self.target_transform(label)

        mel_spec = mel_spec.squeeze(0)

        return mel_spec, label, category, fold

    def get_class_names(self):
        """返回所有类别名称"""
        return sorted(self.metadata['category'].unique())

    def get_class_distribution(self):
        """返回类别分布统计"""
        return self.metadata['category'].value_counts()
