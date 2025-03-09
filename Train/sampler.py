from tqdm import tqdm
import torch
import yaml
import numpy as np
from utils.config_util import AppConfig
from Diffusion.diffusion import get_schedule, extract_into_tensor, corrupt_and_upscale_image
from Loss.losses import calculate_loss_dev, calculate_loss_dev_three_channel, l2_loss, voriticity_residual_three_channel, boundary_condition_residual
with open('configs/config.yml') as f:
    raw_config = yaml.safe_load(f)
config = AppConfig(**raw_config)

def validation(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, data in progress_bar:
            dataset = data.to(device)
            sqrt_beta = get_schedule(config)
            beta = sqrt_beta ** 2
            beta_prev = np.append(0.0, beta[:-1])
            alpha = beta - beta_prev
            posterior_variance = config.diffusion.kappa ** 2 * beta_prev / beta * alpha
            posterior_variance_clipped = np.append(
                posterior_variance[1], posterior_variance[1:]
            )
            posterior_log_variance_clipped = np.log(posterior_variance_clipped)
            posterior_mean_coef1 = beta_prev / beta
            posterior_mean_coef2 = alpha / beta
            y0 = corrupt_and_upscale_image(config, data=dataset).to(device)
            noise = torch.randn(dataset.shape).to(device)
            x_prev = (y0 + config.diffusion.kappa * noise).to(device)
            for i in list(range(config.diffusion.steps))[::-1]:
                x_prev = x_prev.to(device)
                var = extract_into_tensor(posterior_variance_clipped, i, broadcast_shape=dataset.shape)
                sd = np.sqrt(var).to(device)
                coef1 = extract_into_tensor(posterior_mean_coef1, i, broadcast_shape=dataset.shape).to(device)
                coef2 = extract_into_tensor(posterior_mean_coef2, i, broadcast_shape=dataset.shape).to(device)
                t = torch.tensor([i], device=device)

                if i != 0:
                    x_prev = coef1 * x_prev + coef2 * model(x_prev, t) + sd * torch.randn(dataset.shape).to(device)
                else:
                    x_0 = model(x_prev, t)

            loss, loss_mse, loss_adv, loss_dif, loss_bc = calculate_loss_dev(config, x_0, dataset)
            total_loss += loss.item()
            progress_bar.set_description(f'Loss: {loss.item():.4f}'
                                         f'Loss_MSE: {loss_mse.item():.4f}'
                                         f'Loss_DEV: {loss_adv.item():.4f}'
                                         f'Loss_DIF: {loss_dif.item():.4f}'
                                         f'Loss_BC: {loss_bc.item():.4f}')
        return total_loss / len(dataloader)

@torch.no_grad()
def validation_three_channel(model, dataloader, device, config, scaler):
    model.eval()
    total_loss, total_mse_loss, total_pinns_loss, total_bc_loss = 0, 0, 0, 0
    recons = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
    for batch_idx, data in progress_bar:
        # dataset is your x_0
        dataset = data.to(device)                 # shape: [B, C, H, W]
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
        y0 = corrupt_and_upscale_image(config, dataset).to(device)  # shape: [B, C, H, W]
        noise = torch.randn_like(dataset).to(device)                # shape: [B, C, H, W]

        # Initialize x_prev at t = T (the final time step).
        # Typically x_T = y0 + kappa*noise, or the same formula used in forward mode
        x_prev = (y0 + config.diffusion.kappa * noise).to(device)

        # Reverse loop: from t = T-1 down to 0
        for i in reversed(range(config.diffusion.steps)):
            var  = extract_into_tensor(posterior_variance_clipped, i, dataset.shape).to(device)
            sd   = torch.sqrt(var)
            coef1 = extract_into_tensor(posterior_mean_coef1, i, dataset.shape).to(device)
            coef2 = extract_into_tensor(posterior_mean_coef2, i, dataset.shape).to(device)

            # -- Key fix: create a 1D tensor of timesteps for the entire batch
            batch_size = x_prev.shape[0]  # how many images/samples in the batch
            t = torch.full((batch_size,), i, device=device, dtype=torch.float32)

            #print(t)
            if i != 0:
                model_output = model(x_prev, t=t, y0=y0)  # Note the explicit kwarg names
                x_prev = coef1 * x_prev + coef2 * model_output + sd * torch.randn_like(x_prev)
            else:
                x_0 = model(x_prev, t=t, y0=y0)  # Note the explicit kwarg names
                recons.append(x_0.detach().cpu())  # Store reconstruction on CPU to save GPU memory
        # Evaluate your PDE or reconstruction losses on the predicted x_0
        loss, loss_mse, residual, loss_bc = calculate_loss_dev_three_channel(
            config, x_0, dataset, scaler
        )
        total_loss += loss.item()
        total_mse_loss += loss_mse.item()
        total_pinns_loss += residual.item()
        total_bc_loss += loss_bc.item()

        progress_bar.set_description(
            f'Loss: {loss.item():.4f}  '
            f'L2: {loss_mse.item():.4f}  '
            f'Residual: {residual.item():.4f}  '
            f'BC: {loss_bc.item():.4f}'
        )

    avg_val_loss   = total_loss / len(dataloader)
    avg_mse_loss   = total_mse_loss / len(dataloader)
    avg_pinns_loss = total_pinns_loss / len(dataloader)
    avg_bc_loss    = total_bc_loss / len(dataloader)

    print(f'Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Average L2 Loss: {avg_mse_loss:.4f}, '
          f'Average PINNs Loss: {avg_pinns_loss:.4f}, '
          f'Average BC Loss: {avg_bc_loss:.4f}')

    return avg_val_loss, torch.cat(recons, dim=0) 

def validation_unet(model, dataloader, device, config, scaler):
    model.eval()
    total_loss, total_mse_loss, total_pinns_loss, total_bc_loss = 0, 0, 0, 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, data in progress_bar:
            dataset = data.to(device)
            y0 = corrupt_and_upscale_image(config, data=dataset).to(device)
            output = model(y0).to(device)
            loss_mse = l2_loss(scaler.inverse(output), scaler.inverse(dataset))
            residual = voriticity_residual_three_channel(scaler.inverse(output))
            # BC Loss
            left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(scaler.inverse(output))
            loss_bc = l2_loss(left_edge, right_edge) + l2_loss(right_edge, left_edge)
            loss = (0.7 * loss_mse + 0.2 * residual/1000 + 0.005 * loss_bc)
            total_loss += loss.item()
            total_mse_loss += loss_mse.item()
            total_pinns_loss += residual.item()
            total_bc_loss += loss_bc.item()
            progress_bar.set_description(f'Loss: {loss.item():.4f}'
                                         f'Loss_L2: {loss_mse.item():.4f}'
                                         f'Loss_Residual: {residual.item():.4f}'
                                         f'Loss_BC: {loss_bc.item():.4f}')
            avg_val_loss = total_loss / len(dataloader)
            avg_mse_loss = total_mse_loss / len(dataloader)
            avg_pinns_loss = total_pinns_loss / len(dataloader)
            avg_bc_loss = total_bc_loss / len(dataloader)
        
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Average L2 Loss: {avg_mse_loss:.4f},'
              f'Average PINNs Loss: {avg_pinns_loss:.4f}, '
              f'Average BC Loss: {avg_bc_loss:.4f}')
        return(avg_val_loss)
