import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import numpy as np
import torch

def load_flow_data(path, process=None):
    '''
    Load flow data from path [N, T, h, w]
    Flatten the data into shape [N * T, 1, h, w]
    Return the flattened data, mean, and sd
    Load only the specified subset of the data based on the process parameter
    '''
    # Load data
    data = np.load(path)
    data_mean, data_scale = np.mean(data), np.std(data)
    print('Original data shape:', data.shape)

    # Split the data based on the process parameter
    if process == 'train':
        N = int(data.shape[0] * 0.8)  # Use 80% for training
        data = data[:N, ...]
    elif process == 'dev':
        N_start = int(data.shape[0] * 0.8)
        N_end = int(data.shape[0] * 0.9)  # Use next 10% for dev/validation
        data = data[N_start:N_end, ...]
    elif process == 'test':
        N_start = int(data.shape[0] * 0.9)  # Use last 10% for testing
        data = data[N_start:, ...]
    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")
    print(f'Data range: mean: {data_mean}, scale: {data_scale}')
    print(f"data shape for {process} is {data.shape}")
    # Convert data to torch.Tensor and flatten
    data = torch.tensor(data, dtype=torch.float32)  # Use torch.tensor() directly
    N, T, h, w = data.shape
    flattened_data = data.view(N * T, 1, h, w)  # Use view for efficient reshaping

    print(f'Flattened data shape: {flattened_data.shape}')
    return flattened_data, data_mean, data_scale

def load_flow_data_three_channel(path, process=None):
    # load flow data from path
    data = np.load(path)   # [N, T, h, w]
    print('Original data shape:', data.shape)
    data_mean, data_scale = np.mean(data), np.std(data)
    data_min, data_max = np.min(data), np.max(data)  # Added min and max calculation
    
    if process == 'train':
        N = int(data.shape[0] * 0.8)  # Use 80% for training
        data = data[:N, ...]
    elif process == 'dev':
        N_start = int(data.shape[0] * 0.8)
        N_end = int(data.shape[0] * 0.9)  # Use next 10% for dev/validation
        data = data[N_start:N_end, ...]
    elif process == 'test':
        N_start = int(data.shape[0] * 0.9)  # Use last 10% for testing
        data = data[N_start:, ...]
    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")
        
    print(f'Data range: mean: {data_mean}, scale: {data_scale}, min: {data_min}, max: {data_max}')
    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-2):
            flattened_data.append(data[i, j:j+3, ...])
    flattened_data = torch.stack(flattened_data, dim=0)
    print(f'data shape: {flattened_data.shape}')
    
    return flattened_data, data_mean, data_scale, data_min, data_max

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std

class MinMaxScaler(object):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.scale_range = max_val - min_val
    
    def __call__(self, x):
        # Scales to [-1, 1]
        return 2.0 * (x - self.min_val) / self.scale_range - 1.0
    
    def inverse(self, x):
        # Converts back from [-1, 1] to original range
        return (x + 1.0) * self.scale_range / 2.0 + self.min_val
    
    def scale(self):
        # Returns the scaling factor (similar to std in StdScaler)
        return self.scale_range / 2.0


def nearest_torch(data):
    # Keep everything in torch and on GPU
    zero_mask = (data == 0).float()
    non_zero_mask = (data != 0).float()
    
    # Use max pooling with increasing kernel sizes to find nearest non-zero values
    filled_data = data.clone()
    kernel_size = 3
    max_iterations = 10  # Adjust based on your needs
    
    for _ in range(max_iterations):
        # Use max pooling to propagate values
        padded = F.pad(filled_data, (kernel_size//2,)*4, mode='replicate')
        pooled = F.max_pool2d(padded, kernel_size=kernel_size, stride=1, padding=0)
        
        # Update only zero positions
        filled_data = torch.where(zero_mask == 1, pooled, filled_data)
        
        # Check if all zeros are filled
        if not torch.any(filled_data == 0):
            break
            
        kernel_size += 2
    
    return filled_data
    
def corrupt_and_upscale_image(method, data, mode=None, scale=None, portion=None):
    """
    Corrupt and upscale a batch of images.
    """
    assert method in ['skip', 'portion']
    assert mode in ['single', 'multiple']
    
    N, C, H, W = data.shape
    
    # Fast path for skip method using F.interpolate
    if method == 'skip' and mode == 'single':
        assert scale is not None
        down_size = (H//scale, W//scale)
        blur_data = F.interpolate(data, size=down_size, mode='nearest')
        blur_data = blur_data 
        if H == 256:
            return F.interpolate(blur_data, size=(256, 256), mode='nearest')
        elif H == 112:
            return F.interpolate(blur_data, size=(112, 192), mode='nearest')
        elif H == 128:
            return F.interpolate(blur_data, size=(128, 48), mode='nearest')
    # For portion and multiple modes that need griddata
    upscaled = []
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))  # Pre-compute grid once

    if mode == 'multiple' and torch.rand(1).item() < 0.5:
        # Fast path for skip in multiple mode
        scales = [4, 8]
        random_scale = scales[torch.randint(0, len(scales), (1,)).item()]
        down_size = (H//random_scale, W//random_scale)
        blur_data = F.interpolate(data, size=down_size, mode='nearest')
        return F.interpolate(blur_data, size=(256, 256), mode='nearest')

    # Process each image in batch for portion method
    for n in range(N):
        for c in range(C):
            img = data[n, c].cpu().numpy()
            
            if method == 'portion' and mode == 'single':
                assert portion is not None
                num_points = int(H * W * portion)
                idx_x = np.random.randint(0, H, num_points)
                idx_y = np.random.randint(0, W, num_points)
            else:  # multiple mode with portion
                portions = [0.01, 0.05]
                random_portion = portions[torch.randint(0, len(portions), (1,)).item()]
                num_points = int(H * W * random_portion)
                idx_x = np.random.randint(0, H, num_points)
                idx_y = np.random.randint(0, W, num_points)
            
            points = np.column_stack([idx_y, idx_x])
            values = img[idx_x, idx_y]
            interpolated = griddata(points, values, (grid_x, grid_y), method='nearest')
            upscaled.append(interpolated)

    # Convert back to tensor and resize
    upscaled = torch.tensor(upscaled, device=data.device).view(N, C, H, W)
    return F.interpolate(upscaled, size=(256, 256), mode='nearest')
    
def show_blur_image(data, num, chan, n):
    '''
    Display a blurred image and save the plot with minimal white space.

    Parameters:
    - data: A tensor representing the batch of images, with shape [N, C, H, W].
    - num: The index of the image in the batch to display.
    - chan: The channel of the image to display.
    - n: The timestep or identifier for the image being processed.
    '''
    if data.dim() != 4:
        raise ValueError("Expected data to have shape [N, C, H, W]")
    image_data = data[num, chan]
    fig, ax = plt.subplots()  # Use subplots to get more control over layout
    ax.imshow(image_data.cpu().numpy(), cmap='twilight')
    ax.axis('off')  # Turn off axis to remove ticks and labels

    plt.title(n, pad=20)  # Add a title with padding to ensure it's included

    # Convert n to a simple numeric or string value if it's a tensor
    # n_value = n.item() if torch.is_tensor(n) else n
    plt.savefig(f'output_image/{n}.png')
    # Save the plot with minimal white space
    #plt.show()


class FlowDataset(Dataset):
    '''
    Load the dataset
    Normalize the shape
    Get mean and sd
    Finally normalize it
    '''
    def __init__(self, path, process, transform=False):
        # Load data
        self.data, self.mean, self.sd = load_flow_data(path, process)
        # Set the transformation method based on the transform argument
        if transform == 'std':
            self.transform = StdScaler(self.mean, self.sd)  # Assuming StdScaler is a defined class
        elif transform is None:
            self.transform = None  # No transformation will be applied
        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the individual data sample
        data_sample = self.data[idx]

        # Apply transformations, if any
        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample

class FlowDataset_three_channel(Dataset):
    '''
    Load the dataset
    Normalize the shape
    Get mean and sd
    Finally normalize it
    '''
    def __init__(self, path, process, transform=False):
        # Load data
        self.data, self.mean, self.sd, self.data_min, self.data_max = load_flow_data_three_channel(path, process)
        # Set the transformation method based on the transform argument
        if transform == 'std':
            self.transform = StdScaler(self.mean, self.sd)  
        elif transform == 'maxmin':
            self.transform = MinMaxScaler(self.data_min, self.data_max)
        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the individual data sample
        data_sample = self.data[idx]

        # Apply transformations, if any
        if self.transform:
            data_sample = self.transform(data_sample)

        return data_sample


def create_dataloader(path, batch_size=32, shuffle=True, num_workers=0, transform=None):
    # Initialize the dataset
    dataset = FlowDataset(path=path, transform=transform)

    # Create and return the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader




