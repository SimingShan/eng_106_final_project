
import argparse

from torch.utils.data import DataLoader
import numpy as np
import torch
import yaml
from data_utils.data_loader import PairedDataset
from tqdm import tqdm
from Loss.losses import voriticity_residual_three_channel, compute_mre
import matplotlib.pyplot as plt
#from data_utils.LR_mixing import corrupt_and_upscale_image
from Diffusion.diffusion import get_schedule, extract_into_tensor
from data_utils.data_loader import FlowDataset
from Diffusion.u_net_attention import ConditionalModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logger(log_file):
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    return log_message
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace
with open('configs/model_config.yml', 'r') as f:  # Fixed syntax here
    config = yaml.safe_load(f)
config = dict2namespace(config)

def sample_all(model, dataloader, device, config, scaler):
    model.eval()
    all_outputs = []
    all_x0s = []  # New list to store all x0 predictions
    total = 0
    total_2 = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, (lr, data) in progress_bar:
            #plt.imshow(lr.detach().cpu().numpy()[0,0], cmap ='twilight')
            #plt.show()
            collection = []
            for i in range(10):
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

                y0 = lr.to(device)

                noise = torch.randn(dataset.shape).to(device)
                x_prev = (y0 + 2 * noise).to(device)

                for j in list(range(config.diffusion.num_diffusion_steps))[::-1]:
                    x_prev = x_prev.to(device)
                    var = extract_into_tensor(posterior_variance_clipped, j, broadcast_shape=dataset.shape)
                    sd = np.sqrt(var).to(device)
                    log_variance = extract_into_tensor(posterior_log_variance_clipped, j,
                                                       broadcast_shape=[1, 1, 256, 256]).to(device)
                    coef1 = extract_into_tensor(posterior_mean_coef1, j, broadcast_shape=dataset.shape).to(device)
                    coef2 = extract_into_tensor(posterior_mean_coef2, j, broadcast_shape=dataset.shape).to(device)
                    t = torch.tensor([j], device=device)
                    if j != 0:
                        x_prev = coef1 * x_prev + coef2 * model(x_prev, t, y0) + sd * torch.randn(dataset.shape).to(device)
                    else:
                        x_0 = model(x_prev, t, y0)
                collection.append(x_0)
            
            average = torch.mean(torch.stack(collection), axis=0)
            #plt.imshow(average.detach().cpu().numpy()[0,0], cmap ='twilight')
            #plt.show()
            all_x0s.append(average.cpu())  # Store each batch's average prediction
            
            mre = compute_mre(average, dataset)
            pde = voriticity_residual_three_channel(scaler.inverse(average))
            total += mre
            total_2 += pde
            progress_bar.set_description(f"MRE: {mre} pde: {pde}")
            
    #final_tensor = torch.cat(all_outputs, dim=0)
    all_x0s = torch.cat(all_x0s, dim=0)  # Concatenate all predictions
    
    avg_mre = total/len(dataloader)
    avg_pde = total_2/len(dataloader)
    
    print(f'Average MRE: {avg_mre}')
    print(f'Average PDE: {avg_pde}')
    
    return avg_mre, all_x0s  # Return both the average MRE and all predictions

def sample_all_direct(model, dataloader, device, config, scaler):
    model.eval()
    total_loss = 0
    all_outputs = []
    with (torch.no_grad()):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validating Epoch')
        for batch_idx, data in progress_bar:
            dataset = data.to(device)
            y0 = corrupt_and_upscale_image(config, data=dataset).to(device)
            x_0 = model(y0, 0)
            all_outputs.append(scaler.inverse(x_0))
    final_tensor = torch.cat(all_outputs, dim=0)

    return final_tensor
model = ConditionalModel(config).to(device)
mode = 'three'

path_list = ['LR_0.1.npy', 'LR_0.05.npy', 'LR_4_down_nearest.npy', 'LR_8_down_nearest.npy']
save_path = 'output_model_kolmogrov/pird_steps20_kappa2.0_lr0.001.pth'
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
for i in path_list:
    entire_path = 'Datasets/km_data/' + i
    LR = np.load(entire_path)
    HR = np.load('train.npy')
    HR_utils = FlowDataset(HR, 'test',normalization='std')
    scaler = HR_utils.transform
    dataset_test = PairedDataset(corrupted_dataset=LR, hr_dataset=HR, process='test', normalization='std')
    scaler = dataset_test.transform
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)
    sample, x0s= sample_all(model, dataloader_test, device, config, scaler)
    print(sample)
    print(x0s.shape)
    np.save(f'km_4_result_resshift_{i}.npy', x0s.cpu().numpy())

