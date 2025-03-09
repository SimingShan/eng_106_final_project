import torch.nn.functional as F
from scipy.spatial import cKDTree
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def nearest(data):
    # Ensure arr is a numpy array
    arr = data
    arr = np.array(arr)
    
    # Get dimensions for progress tracking
    batch_size, num_channels = arr.shape[0], arr.shape[1]
    
    # Create progress bars for batch and channel processing
    for i in tqdm(range(batch_size), desc='Processing batches'):
        for j in tqdm(range(num_channels), desc=f'Processing channels for batch {i+1}/{batch_size}', leave=False):
            # Extract the 2D slice
            slice_2d = arr[i, j, :, :]
            
            # Find the indices of zero and non-zero elements in the 2D slice
            zero_indices = np.argwhere(slice_2d == 0)
            non_zero_indices = np.argwhere(slice_2d != 0)
            
            # If there are no zero indices or non-zero indices, continue to the next slice
            if zero_indices.size == 0 or non_zero_indices.size == 0:
                continue
                
            # Build a KD-Tree from the non-zero indices for efficient nearest neighbor search
            tree = cKDTree(non_zero_indices)
            
            # For each zero index, find the nearest non-zero index and get its value
            for zero_idx in zero_indices:
                distance, nearest_idx = tree.query(zero_idx)
                nearest_non_zero_idx = non_zero_indices[nearest_idx]
                # Replace the zero value with the nearest non-zero value
                arr[i, j, zero_idx[0], zero_idx[1]] = slice_2d[tuple(nearest_non_zero_idx)]
    
    return arr

def upscale_image(method, data):
    if method == 'portion':
        data = nearest(data)
        data = torch.from_numpy(data).float()
        data = F.interpolate(data, size=(256, 256), mode='nearest')
    else:
        data = torch.from_numpy(data).float()
        data = F.interpolate(data, size=(256, 256), mode='nearest')
    return data

def corrupt_and_upscale_image(config, data):
    print("Starting image corruption and upscaling process...")
    method = config.corruption.method
    data = data
    scale = config.corruption.scale
    portion = config.corruption.portion
    
    if method == 'skip':
        blur_data = data[:, :, ::scale, ::scale]
    elif method == 'average':
        blur_data = torch.nn.functional.avg_pool2d(data, kernel_size=scale, stride=scale, padding=0)
    elif method == 'portion':
        if portion is None:
            raise ValueError("Portion must be specified for the 'portion' method.")
        N, C, H, W = data.shape
        total_pixels = H * W
        pixels_to_keep = int(total_pixels * portion)
        # Create a random mask for the entire batch and all channels at once
        flat_indices = torch.randperm(total_pixels)[:pixels_to_keep]
        mask = torch.zeros((N, C, total_pixels), dtype=torch.bool)
        mask[:, :, flat_indices] = True
        mask = mask.view(N, C, H, W)
        # Apply the mask
        blur_data = data * mask.float().numpy()
    
    print(f"Upscaling images using method: {method}")
    data = upscale_image(method, blur_data)
    print("Processing complete!")
    return data

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace

def main():
    print("Loading configuration...")
    with open('../configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    print("Loading training data...")
    data = np.load('../dataset/train.npy')
    
    print("Starting corruption and upscaling process...")
    LR = corrupt_and_upscale_image(config, data)
    
    print("Saving processed data...")
    np.save('../dataset/LR_0.05.npy', LR)
    print("Process completed successfully!")

if __name__ == '__main__':
    main()