import torch
import csv
from models import resnet18_mfcc_classifier
from utils.data_preprocessing import mp3_to_tensor, waveform_to_mfcc


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_pretrained_model(params_pth, device):
    model = resnet18_mfcc_classifier(in_channels=32, num_classes=65).to(device)
    model.load_state_dict(torch.load(params_pth))
    return model.eval()


def get_first_row(file_path):
    """
    获取CSV文件的首行并返回为列表。

    参数:
        file_path (str): CSV文件的路径。

    返回:
        first_row: CSV文件的首行。
    """
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        first_row = next(reader)  # 获取第一行
        first_row = list(first_row)
    return first_row


def process(waveform, model, genre_list):
    device = get_device()
    mfcc = waveform_to_mfcc(waveform).unsqueeze(0).to(device)
    output = model(mfcc)
    _, max_indices = torch.max(output, dim=1)
    return genre_list[max_indices], max_indices.item(), output


def processor_main(params_pth, mp3_data: str, csv_path):
    device = get_device()
    model = get_pretrained_model(params_pth, device)
    waveform, _ = mp3_to_tensor(mp3_data)
    genre_list = get_first_row(csv_path)
    genre_name, max_idx, output = process(waveform, model, genre_list)
    return genre_name, max_idx, output
