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


def mp3_to_tensor(id_or_file_path: str):
    """
    将MP3音频文件转换为PyTorch张量。

    参数:
        file_path (str): MP3文件的路径。

    返回:
        waveform (torch.Tensor): 音频信号的张量表示。
        sample_rate (int): 采样率。
    """
    # 使用torchaudio加载音频文件

    if id_or_file_path.isdigit():
        mp3_data = crawl_data_by_id(id_or_file_path)
        waveform_mono, sample_rate = load_bytes_data(mp3_data)
    else:
        waveform_mono, sample_rate = load_from_path(id_or_file_path)

    return waveform_mono, sample_rate


def waveform_to_mfcc(waveform, n_mfcc=32, n_fft=2048, n_mels=128, hop_length=1024):
    transforms = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length
    })
    mfcc = transforms(waveform)
    return mfcc
