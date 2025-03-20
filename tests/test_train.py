import torch
from models import ResNetMusicClassifier20
from datasets import TrackDatasetOnline
from torch.utils.data import DataLoader
from utils import collate_fn
from torch.nn import CrossEntropyLoss
from train import Trainer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True

    learning_rate = 0.0001
    num_classes = 90

    model = ResNetMusicClassifier20(num_classes=num_classes)
    train_dataset = TrackDatasetOnline(csv_file_path='../data/v1/train.csv')
    test_dataset = TrackDatasetOnline(csv_file_path='../data/v1/test.csv')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, collate_fn=collate_fn.wave_pad_collate_fn,
                                  num_workers=4, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, collate_fn=collate_fn.wave_pad_collate_fn,
                                 num_workers=4, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model_saving_path = '../checkpoints/version1/'

    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, criterion)
    trainer.train(num_epochs=30, num_epochs_per_test=2, num_epochs_per_plot=4, num_epochs_per_save=5,
                  lr_scheduler_kwargs={'step_size': 5, 'gamma': 0.1}, gradient_accumulation_steps=4,
                  model_saving_path=model_saving_path)


if __name__ == '__main__':
    train()
