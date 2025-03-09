from pydantic import BaseModel
class DatasetConfig(BaseModel):
    path: str
    transform: str
    image_size: int
    image_height: int
    image_width: int

class DiffusionConfig(BaseModel):
    target: str
    sf: int
    schedule_name: str
    etas_end: float
    steps: int
    min_noise_level: float
    kappa: float
    weighted_mse: bool
    predict_type: str
    scale_factor: float
    normalize_input: bool
    latent_flag: bool
    kwargs: float
    num_diffusion_steps: int

class ModelConfig(BaseModel):
    in_channels: int
    out_ch: int
    ch: int
    ch_mult: list
    num_res_blocks: int
    attn_resolutions: tuple
    dropout: float
    resamp_with_conv: bool
    
class TrainConfig(BaseModel):
    epoch: int
    lr: float
    save_interval: int
    batch_size: int
    model_type: str

class AppConfig(BaseModel):
    dataset: DatasetConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    train: TrainConfig
    

