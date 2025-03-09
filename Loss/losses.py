import torch
import numpy as np
import torch
def voriticity_residual_three_channel(w, re=1000.0, dt=1/32):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)
    diffusion = (1.0 / re) * wlap
    residual = wt + (advection - diffusion + 0.1*w[:, 1:-1]) - f
    residual_loss = ((residual ** 2).mean())
    return residual_loss

def compute_rmse(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()

def compute_mse(x, y):
    return ((x - y)**2).mean((-1, -2)).mean()

def compute_mre(pred, target):
    return (((pred[:, 0:1, :, :] - target[:, 0:1, :, :])**2).sum((-1, -2)).sqrt() / 
            ((target[:, 0:1, :, :]**2).sum((-1, -2)).sqrt())).mean()
    
def calculate_loss_three_channel(config, model, xt, x0, t, y0, scaler):
    """
    Calculate the combined loss (RMSE + PDE residual) for a three-channel PDE problem.
    
    :param config:   the config file/dict
    :param model:    the U-Net (or diffusion) model
    :param xt:       the corrupted/noisy version of x0 (shape [N, C, H, W])
    :param x0:       the reference "clean" data (shape [N, C, H, W])
    :param t:        the current time-step (scalar or tensor)
    :param y0:       the degraded/conditional image you want the model to see
    :param scaler:   a scaling object for PDE computations
    
    :return: (loss, loss_mse, residual, loss_bc)
    """
    # ----- 1) Forward the model, now giving y0 -----
    # E.g. model(xt, t, y0=y0) if you adopt that signature
    output = model(x=xt, t=t, y0=y0)

    # ----- 2) PDE + boundary losses just as before -----
    loss_mse  = l2_loss(output, x0)
    residual  = voriticity_residual_three_channel(scaler.inverse(output))
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(output)
    loss_bc   = l2_loss(left_edge, right_edge) + l2_loss(top_edge, bottom_edge)

    # Weighted combination
    loss = 1 * loss_mse #+ 2.0 * residual / 1000.0 + 0.0 * loss_bc
    return loss, loss_mse, residual, loss_bc


def calculate_loss_dev_three_channel(config, output, data, scaler):
    '''
    This function is for calculate the combined loss(RMSE + PINNS) from the model output
    Input:
        config file
        model : u-net
        xt : the corrupted version of x0
        data : HR reference
        t : timestep t
    Output:
        loss : combined loss
    '''
    # Regular MSE
    loss_mse = l2_loss(scaler.inverse(output), scaler.inverse(data))
    residual = voriticity_residual_three_channel(scaler.inverse(output))
    # BC Loss
    left_edge, right_edge, top_edge, bottom_edge = boundary_condition_residual(scaler.inverse(output))
    loss_bc = l2_loss(left_edge, right_edge) + l2_loss(right_edge, left_edge)
    loss = (1 * loss_mse
            + 0 * residual/1000
            + 0 * loss_bc)
    return loss, loss_mse, residual, loss_bc
