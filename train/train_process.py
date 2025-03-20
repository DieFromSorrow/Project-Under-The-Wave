import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(os.pardir)

import torch
import pandas as pd
from models import resnet18_mfcc_classifier, resnet34_mfcc_classifier, \
    eca_resnet18_mfcc_classifier, eca_resnet26_mfcc_classifier, eca_resnet38_mfcc_classifier, \
    attn_resnet18_mfcc_classifier
from datasets import MfccDatesetOnline, MfccDatasetLocal, Augmentor
from torch.utils.data import DataLoader
from utils import collate_fn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR
from train.trainer import Trainer
from utils import running_print

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')  # 修改共享策略


def train_mfcc_classifier(model_name, learning_rate, batch_size, num_epochs, ga_steps, weight_decay,
                          num_workers, data_root=None, data_load_online=False, pretrained_weights=None,
                          num_epochs_per_test=2, num_epochs_per_plot=10, num_epochs_per_save=10):
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
    elif model_name == 'eca_resnet18':
        model = eca_resnet18_mfcc_classifier(in_channels=32, num_classes=num_columns)
    elif model_name == 'eca_resnet26':
        model = eca_resnet26_mfcc_classifier(in_channels=32, num_classes=num_columns)
    elif model_name == 'eca_resnet38':
        model = eca_resnet38_mfcc_classifier(in_channels=32, num_classes=num_columns)
    elif model_name == 'attn_resnet18':
        model = attn_resnet18_mfcc_classifier(in_channels=32, num_classes=num_columns)
    else:
        raise ValueError(f'No such model: {model_name}')

    if pretrained_weights:
        model.train()
        model.load_state_dict(torch.load(pretrained_weights))

    if data_load_online:
        train_dataset = MfccDatesetOnline(csv_file_path=data_root + 'train.csv', cuda=True, transform=Augmentor())
        test_dataset = MfccDatesetOnline(csv_file_path=data_root + 'test.csv', cuda=True)
    else:
        train_dataset = MfccDatasetLocal(root_dir=data_root, mode='train', enable_cache=False, transform=Augmentor())
        test_dataset = MfccDatasetLocal(root_dir=data_root, mode='test', enable_cache=True)

    pad_collate_fn = collate_fn.get_pad_collate_fn(model_name[:4] == 'attn')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn,
                                  num_workers=num_workers, shuffle=True, pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn,
                                 num_workers=num_workers, shuffle=False, pin_memory=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    model_saving_path = '../checkpoints/version1/'
    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, criterion, lr_scheduler=OneCycleLR)
    running_print(f'''
    hyper arguments:
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
                  lr_scheduler_kwargs={'max_lr': learning_rate * 4,
                                       'total_steps': num_epochs * len(train_dataloader) // ga_steps,
                                       'pct_start': 0.3,
                                       'anneal_strategy': 'cos',
                                       'cycle_momentum': True,
                                       'base_momentum': 0.85,
                                       'max_momentum': 0.95},
                  gradient_accumulation_steps=ga_steps, model_saving_path=model_saving_path)


if __name__ == "__main__":
    kwargs_1 = dict(
        model_name='resnet18',
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=8,
        data_root='../data/v2/',
        data_load_online=True
    )

    kwargs_2 = dict(
        model_name='resnet18',
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=12,
        ga_steps=2,
        weight_decay=1e-6,
        num_workers=8,
        data_root='../data/v2/',
        data_load_online=True
    )

    kwargs_3 = dict(
        model_name='resnet34',
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=16,
        data_root='../data/v2/',
        data_load_online=True
    )

    kwargs_4 = dict(
        model_name='resnet34',
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=32,
        ga_steps=4,
        weight_decay=1e-6,
        num_workers=16,
        data_root='../data/v2/',
        data_load_online=True
    )

    kwargs_5 = dict(
        model_name='eca_resnet38',
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=12,
        ga_steps=2,
        weight_decay=1e-6,
        num_workers=4,
        data_root='../data/mini_saved_mfcc/',
        data_load_online=False
    )

    kwargs_6 = dict(
        model_name='eca_resnet18',
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=30,
        ga_steps=4,
        weight_decay=1e-5,
        num_workers=4,
        data_root='../data/mini_saved_mfcc/',
        data_load_online=False,
        pretrained_weights='../checkpoints/version1/2025-3-4-epoch30.pt'
    )

    kwargs_7 = dict(
        model_name='attn_resnet18',
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=30,
        ga_steps=4,
        weight_decay=1e-5,
        num_workers=4,
        data_root='../data/mini_saved_mfcc/',
        data_load_online=False
    )

    train_mfcc_classifier(**kwargs_6)
