import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os


class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=4800, transform=None, num_channels=2, normalize=True):
        self.root_dir = root_dir
        self.target_length = target_length  # Target length in samples
        self.transform = transform
        self.num_channels = num_channels  # Control whether to load single or dual channels
        self.normalize = normalize  # Control whether to normalize the data
        self.data = []

        # Load data paths and labels
        for label in ['0', '1']:
            folder = os.path.join(root_dir, label)
            for file_name in os.listdir(folder):
                if file_name.endswith('.wav'):
                    self.data.append((os.path.join(folder, file_name), int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(file_path)


        # Ensure the waveform has the required number of channels
        if self.num_channels == 1:  # Single channel
            if waveform.shape[0] > 1:  # If more than 1 channel, select the first one
                waveform = waveform[0:1, :]
        elif self.num_channels == 2:  # Double channel
            if waveform.shape[0] == 1:  # If single channel, duplicate it
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:  # If more than 2 channels, take the first 2
                waveform = waveform[:2, :]

        # Trim or pad the waveform to the target length
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # if self.normalize:
        #     # For each channel, subtract the mean and divide by the standard deviation
        #     for i in range(self.num_channels):
        #         waveform[i] = (waveform[i] - waveform[i].mean()) / waveform[i].std()
        # Normalize each channel independently
        if self.normalize:
            # For each channel, subtract the mean and divide by the standard deviation
            for i in range(self.num_channels):
                min_val = waveform[i].min()
                max_val = waveform[i].max()
                waveform[i] = 2 * (waveform[i] - min_val) / (max_val - min_val) - 1

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


def get_dataloader(scene_path, batch_size=32, num_channels=2, normalize=True):
    dataset = AudioDataset(scene_path, num_channels=num_channels, normalize=normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
