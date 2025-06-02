import io
import torchaudio
from utils import TrackCrawler


def crawl_data_by_id(song_id):
    mp3_data = TrackCrawler.crawl_track(song_id)  # bytes_rate=128001
    if mp3_data is None:
        raise ValueError('Data error')
    return mp3_data


def load_bytes_data(data_obj: bytes):
    data_obj = io.BytesIO(data_obj)
    waveform, sample_rate = torchaudio.load(data_obj)
    waveform_mono = waveform.mean(dim=0)
    return waveform_mono, sample_rate


def load_from_path(data_path: str):
    waveform, sample_rate = torchaudio.load(data_path, format="mp3")
    waveform_mono = waveform.mean(dim=0)
    return waveform_mono, sample_rate


def mp3_to_tensor(track_id):
    """
    将MP3音频文件转换为PyTorch张量。

    参数:
        track_id (str / int): MP3文件的路径。

    返回:
        waveform (torch.Tensor): 音频信号的张量表示。
        sample_rate (int): 采样率。
    """
    # 使用torchaudio加载音频文件

    if track_id.isdigit():
        mp3_data = crawl_data_by_id(track_id)
        waveform_mono, sample_rate = load_bytes_data(mp3_data)
    else:
        waveform_mono, sample_rate = load_from_path(track_id)

    return waveform_mono, sample_rate


def normalize(mfcc):
    # 添加归一化操作（每个MFCC系数在时间轴上标准化）
    mfcc_mean = mfcc.mean(dim=-1, keepdim=True)
    mfcc_std = mfcc.std(dim=-1, keepdim=True)
    norm_mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)  # 加小值防止除零
    return norm_mfcc


def waveform_to_mfcc(waveform, device, n_mfcc=32, n_fft=2048,
                     n_mels=128, hop_length=1024, norm=True):
    transforms = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length
    }).to(device)
    waveform = waveform.to(device)
    mfcc = transforms(waveform)
    if norm:
        mfcc = normalize(mfcc)
    return mfcc
