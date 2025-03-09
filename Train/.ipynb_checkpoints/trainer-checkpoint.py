from Diffusion.diffusion import diffusion, get_schedule, extract_into_tensor
from Loss.losses import calculate_loss_three_channel
from tqdm import tqdm
import torch
from utils.datasets import corrupt_and_upscale_image
from Loss.losses import compute_rmse, compute_mse, compute_mre, voriticity_residual_three_channel
import yaml
import numpy as np
import matplotlib.pyplot as plt
def train_diffusion(model, dataloader, optimizer, device, config, scaler, method, mode, scale, portion):
    model.train()
    total_mse_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch')
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        x_t, t, y0 = diffusion(config, data, device, method, mode, scale, portion)
        output = model(x = x_t, t = t, y0 = y0)
        # Caculate losses
        mse_loss = compute_mse(output, data)
        loss = 1 * mse_loss
        loss.backward()
        optimizer.step()
        total_mse_loss += mse_loss.item()
        progress_bar.set_description(f'MSE: {mse_loss.item():.4f},')
        
    return total_mse_loss / len(dataloader)

def validation_diffusion(model, dataloader, device, config, scaler, method, mode, scale, portion):
    model.eval()
    total_mse_loss = 0
    total_rmse_loss = 0
    total_mre_loss = 0
    reconstructed_flows = []  # List to store all reconstructed flows
    uncertainty_maps = []  # List to store uncertainty maps
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, data in progress_bar:
            dataset = data.to(device)                 
            batch_size = dataset.shape[0]
    
            # Rebuild the forward schedule parts
            sqrt_beta = get_schedule(config)
            beta = sqrt_beta ** 2
            beta_prev = np.append(0.0, beta[:-1])
            alpha = beta - beta_prev
    
            # posterior_variance = kappa^2 * (beta_prev/beta) * alpha
            posterior_variance = (config.diffusion.kappa ** 2) * (beta_prev / beta) * alpha
            posterior_variance_clipped = np.append(posterior_variance[1], posterior_variance[1:])
            posterior_log_variance_clipped = np.log(posterior_variance_clipped)
            posterior_mean_coef1 = beta_prev / beta
            posterior_mean_coef2 = alpha / beta
    
            # Create the same y_0 you used in training
            y0 = corrupt_and_upscale_image(method=method, data=dataset, mode=mode, scale=scale, portion=portion).to(device)
            y0 = y0.float()
            
            # Run the model 10 times to collect samples for uncertainty calculation
            num_samples = 10
            x_0_samples = []
            
            for sample_idx in range(num_samples):
                noise = torch.randn_like(dataset).to(device)
                
                # Initialize x_prev at t = T
                x_prev = (y0 + config.diffusion.kappa * noise).to(device)
                # Reverse loop: from t = T-1 down to 0
                for i in reversed(range(config.diffusion.steps)):
                    var = extract_into_tensor(posterior_variance_clipped, i, dataset.shape).to(device)
                    sd = torch.sqrt(var)
                    coef1 = extract_into_tensor(posterior_mean_coef1, i, dataset.shape).to(device)
                    coef2 = extract_into_tensor(posterior_mean_coef2, i, dataset.shape).to(device)
        
                    batch_size = x_prev.shape[0]
                    t = torch.full((batch_size,), i, device=device, dtype=torch.float32)
        
                    if i != 0:
                        model_output = model(x_prev, t=t, y0=y0)
                        x_prev = coef1 * x_prev + coef2 * model_output + sd * torch.randn_like(x_prev)
                    else:
                        x_0 = model(x_prev, t=t, y0=y0)
                #plt.imshow(x_0.detach().cpu().numpy()[0,0], cmap ='twilight')
                #plt.show()
                x_0_samples.append(x_0)
                
            # Stack samples and compute mean and standard deviation
            x_0_stack = torch.stack(x_0_samples)  # shape: [num_samples, batch_size, channels, height, width]
            x_0_mean = torch.mean(x_0_stack, dim=0)  # Average prediction
            x_0_std = torch.std(x_0_stack, dim=0)    # Uncertainty (standard deviation)
            
            # Store the reconstructed flow (using mean prediction)
            reconstructed_flows.append(x_0_mean.cpu())
            # Store the uncertainty map
            uncertainty_maps.append(x_0_std.cpu())
            
            # Compute losses using mean prediction
            mse_loss = compute_mse(x_0_mean, dataset)
            rmse_loss = compute_rmse(x_0_mean, dataset)
            mre_loss = compute_mre(x_0_mean, dataset)

            total_mse_loss += mse_loss.item()
            total_rmse_loss += rmse_loss.item()
            total_mre_loss += mre_loss.item()
            
            progress_bar.set_description(f'MSE Loss: {mse_loss.item():.4f} '
                                       f'RMSE Loss: {rmse_loss.item():.4f} '
                                       f'MRE Loss: {mre_loss.item():.4f} ')
        
        # Calculate average losses
        avg_mse_loss = total_mse_loss / len(dataloader)
        avg_rmse_loss = total_rmse_loss / len(dataloader)
        avg_mre_loss = total_mre_loss / len(dataloader)
        
        # Concatenate all reconstructed flows and uncertainty maps
        all_reconstructed_flows = torch.cat(reconstructed_flows, dim=0)
        all_uncertainty_maps = torch.cat(uncertainty_maps, dim=0)
            
        print(f'Average MSE Loss: {avg_mse_loss:.4f}, '
              f'Average RMSE Loss: {avg_rmse_loss:.4f}, '
              f'Average MRE Loss: {avg_mre_loss:.4f}, ')
        
        return avg_mse_loss, avg_rmse_loss, avg_mre_loss, all_reconstructed_flows, all_uncertainty_maps