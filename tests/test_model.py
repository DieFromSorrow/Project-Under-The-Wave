import time
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torchaudio
import torch
import io
from models import resnet34_mfcc_classifier4test
from utils import TrackCrawler
from datasets import MfccDateset
from torch.utils.data import DataLoader
from utils import collate_fn


def show_tensor_image(tensor: torch.FloatTensor):
    un_loader = transforms.ToPILImage()  # 创建一个将tensor转化为PIL.Image对象的转换器
    image: Image = un_loader(tensor)  # 将张量转化为PIL.Image对象

    image.show()  # 展示图像
    pass


def plot_waveform(_waveform):
    waveform_np = _waveform.numpy()

    # 绘制波形图
    plt.figure(figsize=(512, 2))
    plt.plot(waveform_np[0])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


def plot_waveform_segment(waveform, sample_rate, start_time, end_time):
    # 计算对应的采样点范围
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # 提取所需片段的波形数据
    waveform_segment = waveform[:, start_sample:end_sample]

    # 将波形张量转换为numpy数组
    waveform_np = waveform_segment.numpy()

    # 绘制波形图
    plt.figure(figsize=(256, 4))
    plt.plot(waveform_np[0])
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform (Segment)')
    plt.show()


def test_wave_res_net():
    mp3_data = TrackCrawler.crawl_track('1471185430')  # br=128000
    waveform, sample_date = torchaudio.load(io.BytesIO(mp3_data), format='mp3')
    plot_waveform_segment(waveform, sample_date, start_time=180, end_time=210)
    waveform_mono = torch.mean(waveform, dim=0)
    batch = waveform_mono.unsqueeze(0).unsqueeze(0)
    print(batch.shape)
    model = resnet34_mfcc_classifier4test(num_classes=2)
    output, o_list = model(batch)
    # for o in o_list:
    #     if o.dim() == 3:
    #         show_tensor_image(o)


def test_mfcc_res_net():
    dataset = MfccDateset(csv_file_path='../data/v1/test.csv')
    dataloader = DataLoader(dataset, batch_size=2,
                            collate_fn=collate_fn.mfcc_pad_collate_fn)
    for inputs, labels in dataloader:
        for mfcc in inputs:
            print(mfcc.shape)
            show_tensor_image(mfcc)
        break


if __name__ == '__main__':
    test_mfcc_res_net()
