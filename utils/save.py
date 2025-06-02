import torch
from torch.utils.data.dataloader import DataLoader
from datasets import MfccDatesetOnline
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def save_all(root='../data/EleM-20/'):
    root = Path(root)
    csv_path_train = root / 'train.csv'
    csv_path_test = root / 'test.csv'
    save_path_train = root / 'train'
    save_path_test = root / 'test'
    save_path_train.mkdir(exist_ok=True)
    save_path_test.mkdir(exist_ok=True)
    dataset_train = MfccDatesetOnline(csv_file_path=csv_path_train, cuda=True)
    dataset_test = MfccDatesetOnline(csv_file_path=csv_path_test, cuda=True)

    dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)
    metadata = []
    with torch.no_grad():
        for idx, (batch, label) in tqdm(enumerate(dataloader)):
            filename = f'{idx:05d}.pt'
            pth = save_path_test / filename
            batch = batch.squeeze(0)
            label = label.squeeze(0)
            sample = (batch, label)
            torch.save(sample, pth)
            metadata.append((filename, label.item()))
        meta_df = pd.DataFrame(metadata, columns=["filename", "label"])
        meta_df.to_csv(root / "metadata_test.csv", index=False)

        dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=8)
        metadata = []
        for idx, (batch, label) in tqdm(enumerate(dataloader)):
            filename = f'{idx:05d}.pt'
            pth = save_path_train / filename
            batch = batch.squeeze(0)
            label = label.squeeze(0)
            sample = (batch, label)
            torch.save(sample, pth)
            metadata.append((filename, label.item()))
        meta_df = pd.DataFrame(metadata, columns=["filename", "label"])
        meta_df.to_csv(root / "metadata_train.csv", index=False)


if __name__ == '__main__':
    save_all()
