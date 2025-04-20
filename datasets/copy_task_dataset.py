import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class CopyTaskDataset(Dataset):
    def __init__(self, signal_length = 10, blank_length=100, samples=10000, device='cuda'):
        super(CopyTaskDataset, self).__init__()
        self.device = device
        self.samples = samples
        self.seq_length = blank_length + 2 * signal_length - 1
        self.blank_length = blank_length
        self.data = torch.zeros((samples, self.seq_length))
        self.target = torch.zeros((samples, self.seq_length))
        blank = [8]
        delimiter = [9]
        for i in range(samples):
            blanks = torch.tensor(blank*(blank_length-1)+delimiter+blank*9)
            signal = torch.randint(0, 8, (signal_length,))
            self.data[i,:] = torch.cat((signal, blanks))
            self.target[i,:] = torch.cat((torch.tensor(blank*(blank_length+9)), signal))
        self.data = self.data.long().to(device)
        self.target = self.target.to(device)

    def __len__(self):
        return self.samples

    def __getitem__(self, item):
        targets = self.data[item, :].unsqueeze(-1)
        labels = self.target[item, :].unsqueeze(-1)
        return targets, labels
