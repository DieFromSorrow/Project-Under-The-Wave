import torch
from torch.utils.data import Dataset
from datasets import TrackDatasetOnline
from utils.data_preprocessing import waveform_to_mfcc
from pathlib import Path
import pandas as pd
import numpy as np


class MfccDatesetOnline(TrackDatasetOnline):
    def __init__(self, csv_file_path, cuda=True, transform=None):
        super(MfccDatesetOnline, self).__init__(csv_file_path)
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else:
            self.device = torch.device('cpu')
            self.cuda = False
        self.transform = transform
        pass

    def __getitem__(self, index):
        waveform, label = super().__getitem__(index)
        mfcc = self.waveform2mfcc(waveform)
        if self.transform:
            mfcc = self.transform(mfcc)
        return mfcc, label

    def waveform2mfcc(self, waveform, n_mfcc=32, n_fft=2048, n_mels=128, hop_length=1024):
        waveform = waveform.to(self.device)
        mfcc = waveform_to_mfcc(waveform, self.device, n_mfcc, n_fft, n_mels,
                                hop_length, norm=True).transpose(0, 1)
        return mfcc


class MfccDatasetLocal(Dataset):
    def __init__(self, root_dir, mode, enable_cache=True, max_cache_size=1000, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录路径
            enable_cache (bool): 是否启用内存缓存（默认True）
            max_cache_size (int): 最大缓存条目数（默认1000）
        """
        if mode != 'train' and mode != 'test':
            raise ValueError(f'Mode must be train or test, but get {mode}')
        self.mode = mode
        self.root = Path(root_dir)
        self.meta_path = self.root / f"metadata_{self.mode}.csv"

        # 读取元数据并验证完整性
        self._validate_metadata()
        self.meta = pd.read_csv(self.meta_path)

        # 初始化缓存系统
        self.enable_cache = enable_cache
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = max_cache_size

        # 预计算统计信息
        self._lengths = []
        self._precompute_stats()

        self.transform = transform

    def _validate_metadata(self):
        """验证元数据文件存在性"""
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.meta_path}. "
                "Please check dataset directory structure."
            )

    def _precompute_stats(self):
        """预计算统计信息用于分析"""
        for idx in range(len(self.meta)):
            filename = self.meta.iloc[idx]['filename']
            filepath = self.root / f"{self.mode}" / filename
            if not filepath.exists():
                raise FileNotFoundError(f"MFCC file {filepath} not found")
            data = torch.load(filepath)
            self._lengths.append(data[0].shape[1])  # 假设MFCC形状为 (n_mfcc, time)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        # 参数校验
        if index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self) - 1}]")

        # 从缓存获取
        if self.enable_cache and index in self.cache:
            return self.cache[index]

        # 加载数据
        filename, _ = self.meta.iloc[index]
        file_path = self.root / f"{self.mode}" / filename

        try:
            mfcc_tensor, label = torch.load(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading {file_path}: {str(e)}"
            ) from e

        # 格式验证
        if not isinstance(mfcc_tensor, torch.Tensor):
            raise ValueError(
                f"Invalid data format in {file_path}: "
                "Expected torch.Tensor, got {type(mfcc_tensor)}"
            )

        # 更新缓存
        if self.enable_cache:
            self._update_cache(index, (mfcc_tensor, label))

        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)

        return mfcc_tensor, label

    def _update_cache(self, index, data):
        """LRU缓存更新策略"""
        if index in self.cache:
            self.cache_order.remove(index)
        else:
            if len(self.cache) >= self.max_cache_size:
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]

        self.cache[index] = data
        self.cache_order.append(index)

    @property
    def mean_length(self):
        """获取平均序列长度"""
        return np.mean(self._lengths)

    @property
    def length_std(self):
        """获取长度标准差"""
        return np.std(self._lengths)

    def analyze_dataset(self):
        """打印数据集统计信息"""
        print(f"Dataset Summary:")
        print(f"Total samples: {len(self)}")
        print(f"MFCC shape: {self[0][0].shape}")  # 假设第一个样本存在
        print(f"Average length: {self.mean_length:.2f} ± {self.length_std:.2f} frames")
        print(f"Label distribution:\n{self.meta['label'].value_counts()}")
