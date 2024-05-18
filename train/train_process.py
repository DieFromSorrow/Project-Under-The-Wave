import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.pardir)

import torch
import pandas as pd
from models import resnet18_mfcc_classifier, resnet34_mfcc_classifier
from datasets import MfccDateset
from torch.utils.data import DataLoader
from utils import collate_fn
from torch.nn import CrossEntropyLoss
from train.trainer import Trainer
from utils import running_print


def train_mfcc_classifier(model_name, learning_rate, batch_size, num_epochs, ga_steps, weight_decay,
                          num_workers, data_root=None,
                          num_epochs_per_test=1, num_epochs_per_plot=2, num_epochs_per_save=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True

    if data_root is None:
        data_root = '../data/v1/'
    elif not (data_root[-1] == '\\' or data_root[-1] == '/'):
        data_root += '/'

    # 读取CSV文件
    df = pd.read_csv(data_root + 'track_lists.csv')
    # 获取列数
    num_columns = len(df.columns)

    if model_name == 'resnet18':
        model = resnet18_mfcc_classifier(in_channels=32, num_classes=num_columns)
    elif model_name == 'resnet34':
        model = resnet34_mfcc_classifier(in_channels=32, num_classes=num_columns)
    else:
        raise ValueError(f'No such model: {model_name}')

    train_dataset = MfccDateset(csv_file_path=data_root + 'train.csv', cuda=False)
    test_dataset = MfccDateset(csv_file_path=data_root + 'test.csv', cuda=False)
    pad_collate_fn = collate_fn.mfcc_pad_collate_fn
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn,
                                  num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn,
                                 num_workers=num_workers, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    model_saving_path = '../checkpoints/version1/'
    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, criterion)
    running_print(f'''hyper arguments:
    model: {model_name}
    learning rate: {learning_rate}
    batch size: {batch_size}
    num epochs: {num_epochs}
    gradient accumulation steps: {ga_steps}
    weight decay: {weight_decay}
    num workers: {num_workers}
    data dictionary: {data_root}
    num classes: {num_columns}''')
    trainer.train(num_epochs=num_epochs, num_epochs_per_test=num_epochs_per_test,
                  num_epochs_per_plot=num_epochs_per_plot, num_epochs_per_save=num_epochs_per_save,
                  lr_decay_kwargs={'step_size': 5, 'gamma': 0.1}, gradient_accumulation_steps=ga_steps,
                  model_saving_path=model_saving_path)


if __name__ == "__main__":
    kwargs_1 = dict(
        model_name='resnet18',
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=8,
        data_root='../data/v2/'
    )

    kwargs_2 = dict(
        model_name='resnet18',
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=12,
        ga_steps=2,
        weight_decay=1e-4,
        num_workers=8,
        data_root='../data/v2/'
    )

    kwargs_3 = dict(
        model_name='resnet34',
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=16,
        data_root='../data/v2/'
    )

    kwargs_4 = dict(
        model_name='resnet34',
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=16,
        data_root='../data/v2/'
    )

    train_mfcc_classifier(**kwargs_2)
