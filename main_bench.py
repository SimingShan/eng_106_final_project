import argparse
from torch.utils.data import DataLoader
import yaml
from utils.config_util import AppConfig
from Diffusion.u_net_attention import ConditionalModel
from utils.datasets import FlowDataset, FlowDataset_three_channel
from Train.train_bench import train_bench, validation_bench
import torch
from datetime import datetime
import os 
from neuralop.models import FNO
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train bench model')
    parser.add_argument('--method', type=str, choices=['skip', 'portion'], default='skip',
                      help='Method for data processing')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single',
                      help='Mode of operation')
    parser.add_argument('--scale', type=int, default=4,
                      help='Scale factor for skip method')
    parser.add_argument('--portion', type=float, default=None,
                      help='Portion value for portion method')
    parser.add_argument('--model_type', type=str, default='unet',
                      help='model used')
    parser.add_argument('--dataset', type=str, default='km',
                      help='which data')
    return parser.parse_args()

def load_config(path='configs/model_config.yml'):
    with open(path) as f:
        raw_config = yaml.safe_load(f)
    return AppConfig(**raw_config)

def setup_logger(log_file):
    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
    return log_message

def get_run_name(args):
    if args.method == 'skip':
        return f"{args.method}_scale{args.scale}"
    else:  # portion
        portion_str = f"portion{args.portion}" if args.portion is not None else "portion_random"
        return f"{args.method}_{portion_str}"

def main():
    args = parse_args()
    dataset_type = args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(f'configs/model_config_{dataset_type}.yml')
    model_type = args.model_type
    if model_type == 'unet':
        model = ConditionalModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    elif model_type == 'fno':
        model = FNO(n_modes=(32, 32), hidden_channels=64,
                in_channels=1, out_channels=1).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.train.epoch, 
        eta_min=0.00001
    )

    # Create run name based on arguments
    run_name = get_run_name(args)
    
    ### Setup for the log file ###
    log_dir = f"log_file/{dataset_type}"
    os.makedirs(log_dir, exist_ok=True)
    #current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"log_{run_name}_{model_type}.txt")
    log = setup_logger(log_file)
    
    ### Prepare the dataset ###
    dataset = FlowDataset(path=config.dataset.path, process='train', transform=config.dataset.transform)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    scaler = dataset.transform
    dataset_dev = FlowDataset(path=config.dataset.path, process='dev', transform=config.dataset.transform)
    dataloader_dev = DataLoader(dataset_dev, batch_size=config.train.batch_size, shuffle=True)
    dataset_test = FlowDataset(path=config.dataset.path, process='test', transform=config.dataset.transform)
    dataloader_test = DataLoader(dataset_test, batch_size=config.train.batch_size, shuffle=False)
    log(f"Starting training with configuration: {run_name}")
    
    for epoch in range(config.train.epoch):
        avg_mse = train_bench(model, dataloader, optimizer, device, scaler, 
                                    method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
        log(f'Epoch: {epoch}, Avg MSE Training Loss: {avg_mse}')
        if (epoch+1) % 100 == 0:
            avg_mse_loss, avg_rmse_loss, avg_mre_loss, recons = validation_bench(
                model, dataloader_dev, device, scaler, 
                method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
            log(f'Epoch: {epoch}, Avg MSE Validation Loss: {avg_mse_loss}, Avg RMSE Validation Loss: {avg_rmse_loss}, '
                f'Avg MRE Validation Loss: {avg_mre_loss}')
        
        # Save checkpoint with descriptive name
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'args': args
        }
        
        os.makedirs("output_model", exist_ok=True)
        save_path = f"output_model/{dataset_type}/{model_type}_{run_name}.pth"
        torch.save(checkpoint, save_path)
        log(f'Checkpoint saved to {save_path}')
        
        scheduler.step()
    avg_mse_loss, avg_rmse_loss, avg_mre_loss, recons = validation_bench(
                model, dataloader_test, device, scaler, 
                method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
    log(f'Epoch: {epoch}, Avg Test MSE Validation Loss for {model_type} at {run_name}: {avg_mse_loss}, Avg RMSE Test Validation Loss for {model_type} at {run_name}: {avg_rmse_loss}, '
        f'Avg Test MRE Validation Loss for {model_type} at {run_name}: {avg_mre_loss}')
    np.save(f"output_sample/{dataset_type}/{model_type}_{run_name}.npy", recons.detach().cpu().numpy())

if __name__ == '__main__':
    main()