import io
import csv
import torch
import random
import torchaudio
from utils.crawler import TrackCrawler
from torch.utils.data import Dataset
from utils import error_print, warning_print
from utils.auto_connect import auto_connect


class TrackDatasetOnline(Dataset):
    def __init__(self, csv_file_path):
        self.data_list = []
        self.csv_to_list(csv_file_path)
        pass

    def csv_to_list(self, csv_file_path):
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # 获取列别名

            for row in reader:
                for index, value in enumerate(row):
                    category_index = int(index % len(headers))  # 计算类别索引值
                    self.data_list.append((int(value), category_index))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        while True:
            data = self.data_list[index]
            try:
                mp3_data = TrackCrawler.crawl_track(data[0])  # bytes_rate=128001
                if mp3_data is None:
                    raise ValueError('Data error')
                mp3_data = io.BytesIO(mp3_data)
                waveform, sample_rate = torchaudio.load(mp3_data)
                waveform_mono = waveform.mean(dim=0)
                label = torch.tensor([data[1]], dtype=torch.int64)
            except Exception as e:
                warning_print(f'There is an EXCEPTION getting or processing data but process CONTINUE, '
                              f'index {index} loss.')
                index = random.randint(0, len(self) - 1)
                warning_print(f'Index {index} instead')
                error_print(e)
            else:
                break
        return waveform_mono, label
