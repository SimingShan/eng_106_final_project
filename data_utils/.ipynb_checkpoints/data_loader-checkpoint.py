import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_utils.data_transformation import StdScaler

def load_flow_data(data, process=None):
    data_mean, data_scale = np.mean(data), np.std(data)
    if process == 'train':
        N = int(data.shape[0] * 0.8)
        data = data[:N, ...]
    elif process == 'dev':
        N_start = int(data.shape[0] * 0.8)
        N_end = int(data.shape[0] * 0.9)
        data = data[N_start:N_end, ...]
    elif process == 'test':
        N_start = int(data.shape[0] * 0.9)
        data = data[N_start:, ...]
    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")
    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    if len(data.shape) == 4:
        for i in range(data.shape[0]):
            for j in range(data.shape[1] - 2):
                flattened_data.append(data[i, j:j + 3, ...])
    elif len(data.shape) == 3:
        for i in range(data.shape[0] - 2):
            flattened_data.append(data[i:i+3, ...])
    else:
        raise ValueError("dataset has the wrong shape(expected shape are 3 and 4)")
    flattened_data = torch.stack(flattened_data, dim=0)
    return flattened_data, data_mean, data_scale


class FlowDataset(Dataset):
    def __init__(self, data, process, normalization=False):
        self.data, self.mean, self.sd = load_flow_data(data, process)
        if normalization:
            self.transform = StdScaler(self.mean, self.sd)
        elif not normalization:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample


def create_dataloader(config, process=None, normalization=False):
    data = np.load(config.dataset.data_path)
    batch_size = config.Training.batch_size
    dataset = FlowDataset(data, process, normalization)
    scaler = dataset.transform
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader, scaler

class PairedDataset(Dataset):
    def __init__(self, corrupted_dataset, hr_dataset, process, normalization):
        assert corrupted_dataset.shape == hr_dataset.shape, "Datasets must be the same size"
        self.hr_dataset, self.mean, self.sd = load_flow_data(hr_dataset, process)
        self.corrupted_dataset, _, _ = load_flow_data(corrupted_dataset, process)
        if normalization == 'std':
            self.transform = StdScaler(self.mean, self.sd)
        elif normalization is None:
            self.transform = None
        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __getitem__(self, index):
        corrupted_data = self.corrupted_dataset[index]
        hr_data = self.hr_dataset[index]
        if self.transform:
            corrupted_data = self.transform(corrupted_data)
            hr_data = self.transform(hr_data)
        return corrupted_data, hr_data

    def __len__(self):
        return len(self.corrupted_dataset)