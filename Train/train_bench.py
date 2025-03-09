from Diffusion.diffusion import diffusion
from Loss.losses import calculate_loss_three_channel
from tqdm import tqdm
import torch
from utils.datasets import corrupt_and_upscale_image
from Loss.losses import compute_rmse, compute_mse, compute_mre, voriticity_residual_three_channel
import matplotlib.pyplot as plt
from neuralop.models import FNO

def train_bench(model, dataloader, optimizer, device, scaler, method = 'skip', mode = 'single', scale = 4, portion = None):
    model.train()
    total_mse_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch')
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        y0 = corrupt_and_upscale_image(method = method, data = data, mode = mode, scale = scale, portion = portion)
        y0 = y0.float()
        #plt.imshow(y0[0,0].detach().cpu().numpy(), cmap = 'twilight')
        #plt.show()
        output = model(y0)
        #plt.imshow(output[0,0].detach().cpu().numpy(), cmap = 'twilight')
        #plt.show()
        # Caculate losses
        mse_loss = compute_mse(output, data)
        # BC Loss
        loss = mse_loss
        loss.backward()
        optimizer.step()
        total_mse_loss += mse_loss.item()
        progress_bar.set_description(f'MSE: {mse_loss.item():.4f},')
        
    return total_mse_loss / len(dataloader)

def validation_bench(model, dataloader, device, scaler, method = 'skip', mode = 'single', scale = 4, portion = None):
    model.eval()
    total_mse_loss = 0
    total_rmse_loss = 0
    total_mre_loss = 0
    reconstructed_flows = [] 
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, data in progress_bar:
            dataset = data.to(device)
            y0 = corrupt_and_upscale_image(method = method, data = dataset, mode = mode, scale = scale, portion = portion).to(device)
            y0 = y0.float()
            #plt.imshow(y0[0,0].detach().cpu().numpy(), cmap = 'twilight')
            #plt.show()
            output = model(y0).to(device)
            #plt.imshow(output[0,0].detach().cpu().numpy(), cmap = 'twilight')
            #plt.show()
            reconstructed_flows.append(output.cpu())
            mse_loss = compute_mse(scaler.inverse(output), scaler.inverse(dataset))
            rmse_loss = compute_rmse(scaler.inverse(output), scaler.inverse(dataset))
            mre_loss = compute_mre(scaler.inverse(output), scaler.inverse(dataset))
            total_mse_loss += mse_loss.item()
            total_rmse_loss += rmse_loss.item()
            total_mre_loss += mre_loss.item()
            progress_bar.set_description(f'MSE Loss: {mse_loss.item():.4f}'
                                         f'RMSE Loss: {rmse_loss.item():.4f}'
                                         f'MRE Loss: {mre_loss.item():.4f}')
            avg_mse_loss = total_mse_loss / len(dataloader)
            avg_rmse_loss = total_rmse_loss / len(dataloader)
            avg_mre_loss = total_mre_loss / len(dataloader)

        # Concatenate all reconstructed flows
        all_reconstructed_flows = torch.cat(reconstructed_flows, dim=0)
        print(f'Average MSE Loss: {avg_mse_loss:.4f},'
              f'Average RMSE Loss: {avg_rmse_loss:.4f}, '
              f'Average MRE Loss: {avg_mre_loss:.4f}, ')
        return avg_mse_loss, avg_rmse_loss, avg_mre_loss, all_reconstructed_flows

