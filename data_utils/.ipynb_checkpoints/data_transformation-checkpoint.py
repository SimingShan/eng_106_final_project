import torch.nn.functional as F
from scipy.spatial import cKDTree
import numpy as np
import torch
#from Evaluation_utils.flow_plot import show_image, show_image_sparse, show_image_combined, show_sparse

import random
#random.seed(3407)
#torch.manual_seed(3407)
def upscale_image(data, target_width, target_height, method):
    if method == 'sparse':
        data = F.interpolate(data, size=(target_width, target_height), mode='nearest')
    else:
        data = F.interpolate(data, size=(target_width, target_height), mode='nearest')
    return data

def corrupt_and_upscale_image(data, config):
    method = config.dataset.corrupt_method
    if method == 'skip':
        scale = config.dataset.corrupt_scale
        downsampled_data = data[:, :, ::scale, ::scale]
    elif method == 'sparse':
        sparsity = config.dataset.sparsity
        N, C, H, W = data.shape
        total_pixels = H * W
        pixels_to_keep = int(total_pixels * sparsity)
        flat_indices = torch.randperm(total_pixels)[:pixels_to_keep]
        mask = torch.zeros((N, C, total_pixels), dtype=torch.bool)
        mask[:, :, flat_indices] = True
        mask = mask.view(N, C, H, W)
        sparse_data = data * mask.float()
        downsampled_data = tessellation(sparse_data)
        downsampled_data = torch.from_numpy(downsampled_data).float()

    upscaled_data = upscale_image(downsampled_data, config.dataset.target_width, config.dataset.target_height, method)
    return upscaled_data

def relative_error_loss(x, y):
    if len(x.shape) == 4:
        numerator = ((x[:, 1:2, :, :] - y[:, 1:2, :, :])**2).sum(dim=(-1, -2)).sqrt()
        denominator = (y[:, 1:2, :, :]**2).sum(dim=(-1, -2)).sqrt()
    relative_error = numerator / denominator
    return relative_error.mean()

def corrupt_and_upscale_image_testing(data, config, method=None, scale=None, sparsity=None, noise=False, uneven_sparse=False):
    noise_level = noise
    #print(noise_level)
    local_generator = torch.Generator()
    local_generator.manual_seed(3407)  # Set a seed for the generator
    if method == 'skip':
        downsampled_data = data[:, :, ::scale, ::scale]
    elif method == 'sparse':
        N, C, H, W = data.shape
        total_pixels = H * W

        if uneven_sparse:
            # Create a probability map for uneven sparsity
            x_axis = torch.linspace(0, 1, steps=W, dtype=torch.float32).pow(0.5).unsqueeze(0).repeat(H,1)  # Smoother gradient
            y_axis = torch.linspace(0, 1, steps=H, dtype=torch.float32).pow(0.5).unsqueeze(1).repeat(1,W)  # Smoother gradient
            probability_map = (x_axis + y_axis) / 2
            probability_map = probability_map / probability_map.sum()

            # Flatten the probability map for sampling
            flat_probability_map = probability_map.flatten()

            # Determine number of points to keep based on sparsity
            pixels_to_keep = int(total_pixels * sparsity)

            # Sample pixels based on the probability map
            flat_indices = torch.multinomial(flat_probability_map, pixels_to_keep, replacement=False)
        else:
            pixels_to_keep = int(total_pixels * sparsity)
            flat_indices = torch.randperm(total_pixels, generator=local_generator)[:pixels_to_keep]

        # Create mask based on sampled indices
        mask = torch.zeros(total_pixels, dtype=torch.bool)
        mask[flat_indices] = True
        mask = mask.view(H, W).unsqueeze(0).unsqueeze(0).repeat(N, C, 1, 1).to(data.device)

        sparse_data = data * mask.float()
        #show_sparse(sparse_data, show_image=True, num=0, save_image=True, title='sparse_point')
        noise_tensor = noise_level * torch.randn(sparse_data.size(), generator=local_generator).to(data.device)
        #show_sparse(sparse_data + noise_tensor * mask.float(), show_image=True, num=0, save_image=True, title='sparse_point_noise')
        downsampled_data = tessellation(sparse_data + noise_tensor * mask.float())
        downsampled_data = torch.from_numpy(downsampled_data).float()
    if noise is not None:
        #print('adding noise ...')
        #show_image(downsampled_data, show_image=True, save_image=True, title='downsampled', num=0)
        #noise_tensor = noise_level * torch.randn(downsampled_data.size(), generator=local_generator).to(downsampled_data.device)
        #downsampled_data = downsampled_data + noise_tensor
        #show_image(downsampled_data, show_image=True, num=0)
        upscaled_data = torch.nn.functional.interpolate(downsampled_data, size=(config.dataset.target_height, config.dataset.target_width), mode='nearest').to(data.device)
    else:
        upscaled_data = torch.nn.functional.interpolate(downsampled_data, size=(config.dataset.target_height, config.dataset.target_width), mode='nearest').to(data.device)

    return upscaled_data
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

def tessellation(data):
    # Ensure arr is a numpy array
    arr = data.cpu().detach()
    arr = np.array(arr)

    # Process each 2D slice individually
    for i in range(arr.shape[0]):  # Iterate over batch
        for j in range(arr.shape[1]):  # Iterate over channels
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

