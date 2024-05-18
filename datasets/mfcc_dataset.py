import torch
import torchaudio
from datasets import TrackDataset


class MfccDateset(TrackDataset):
    def __init__(self, csv_file_path, cuda=True):
        super(MfccDateset, self).__init__(csv_file_path)
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else:
            self.device = torch.device('cpu')
            self.cuda = False
        pass

    def __getitem__(self, index):
        waveform, label = super().__getitem__(index)
        mfcc = self.waveform2mfcc(waveform)
        return mfcc, label

    def waveform2mfcc(self, waveform, n_mfcc=32, n_fft=2048, n_mels=128, hop_length=1024):
        waveform = waveform.to(self.device)
        transforms = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length
            }).to(self.device)
        mfcc = transforms(waveform).transpose(0, 1)
        return mfcc
