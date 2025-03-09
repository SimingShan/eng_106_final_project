import argparse
from torch.utils.data import DataLoader
import yaml
from utils.config_util import AppConfig
from Diffusion.u_net_attention import ConditionalModel
from utils.datasets import FlowDataset, FlowDataset_three_channel
from Train.trainer import train_diffusion, validation_diffusion
import torch
from datetime import datetime
import os 
import numpy as np
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def parse_args():
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--method', type=str, choices=['skip', 'portion'], default='skip',
                      help='Method for data processing')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single',
                      help='Mode of operation')
    parser.add_argument('--scale', type=int, default=4,
                      help='Scale factor for skip method')
    parser.add_argument('--portion', type=float, default=None,
                      help='Portion value for portion method')
    parser.add_argument('--patience', type=int, default=7,
                      help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--val-frequency', type=int, default=100,
                      help='Frequency of validation (every N epochs)')
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
        return f"{args.method}_mode{args.mode}_scale{args.scale}"
    else:  # portion
        portion_str = f"portion{args.portion}" if args.portion is not None else "portion_random"
        return f"{args.method}_mode{args.mode}_{portion_str}"
        
def main():
    args = parse_args()
    dataset_type = args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(f'configs/model_config_{dataset_type}.yml')
    model = ConditionalModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.train.epoch, 
        eta_min=0.00002
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Create run name based on arguments
    run_name = get_run_name(args)
    
    # Setup logging
    log_dir = f"log_file/{dataset_type}"
    os.makedirs(log_dir, exist_ok=True)
    #current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"log_steps{config.diffusion.steps}_kappa{config.diffusion.kappa}_lr{config.train.lr}_{run_name}.txt")
    log = setup_logger(log_file)

    # Setup data
    dataset = FlowDataset(path=config.dataset.path, process='train', transform=config.dataset.transform)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    scaler = dataset.transform
    dataset_dev = FlowDataset(path=config.dataset.path, process='dev', transform=config.dataset.transform)
    dataloader_dev = DataLoader(dataset_dev, batch_size=config.train.batch_size, shuffle=True)
    dataset_test = FlowDataset(path=config.dataset.path, process='test', transform=config.dataset.transform)
    dataloader_test = DataLoader(dataset_test, batch_size=config.train.batch_size, shuffle=False)
    best_mre_loss = float('inf')
    last_val_mre = float('inf')  # Store last validation MRE for early stopping

    for epoch in range(config.train.epoch):
        # Training phase
        avg_mse = train_diffusion(model, dataloader, optimizer, device, config, scaler, 
                                    method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
        log(f'Epoch: {epoch}, Avg MSE Training Loss: {avg_mse}')

        # Validation phase - only every N epochs
        if (epoch+1) % args.val_frequency == 0:
            avg_mse_loss, avg_rmse_loss, avg_mre_loss,  _, _ = validation_diffusion(
                model, dataloader_dev, device, config, scaler, 
                method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
            log(f'Epoch: {epoch}, Avg MSE Validation Loss: {avg_mse_loss}, Avg RMSE Validation Loss: {avg_rmse_loss}, '
            f'Avg MRE Validation Loss: {avg_mre_loss}')
            
            last_val_mre = avg_mre_loss

            # Early stopping check
            if early_stopping(avg_mre_loss):
                log("Early stopping triggered due to no improvement in MRE!")
                break

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'args': args
        }
        
        os.makedirs("output_model", exist_ok=True)
        save_path = f"output_model/{dataset_type}/diffusion_steps{config.diffusion.steps}_kappa{config.diffusion.kappa}_lr{config.train.lr}_{run_name}.pth"
        torch.save(checkpoint, save_path)
        log(f'Checkpoint saved to {save_path}')
                
        scheduler.step()
    avg_mse_loss, avg_rmse_loss, avg_mre_loss, recons, uncertainty = validation_diffusion(
                model, dataloader_test, device, config, scaler, 
                method=args.method, mode=args.mode, scale=args.scale, portion=args.portion)
    log(f'Epoch: {epoch}, Avg MSE test Loss: {avg_mse_loss}, Avg RMSE test Loss: {avg_rmse_loss}, '
    f'Avg MRE test Loss: {avg_mre_loss}')

    np.save(f"output_sample/{dataset_type}/diffusion_{run_name}.npy", recons.detach().cpu().numpy())
    np.save(f"output_sample/{dataset_type}/diffusion_{run_name}_uncertainty.npy", uncertainty.detach().cpu().numpy())

if __name__ == '__main__':
    main()