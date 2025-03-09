import torch
import torch.nn as nn
import math

###############################################
# Supporting functions/classes assumed present:
###############################################
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """Generate sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            # pad for 'circular' if needed
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="circular")
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                  padding_mode='circular')

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='circular')

        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, padding_mode='circular')

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                               stride=1, padding=1, padding_mode='circular')
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                              stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Handle rectangular inputs
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)                  # b,c,hw
        w_ = torch.bmm(q, k)                        # b,hw,hw
        w_ = w_ * (c**-0.5)
        w_ = nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_)     # b,c,hw
        h_ = h_.reshape(b, c, h, w)
        return x + self.proj_out(h_)

###############################################
# ConditionalModel Definition
###############################################
class ConditionalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ch            = config.model.ch
        out_ch        = config.model.out_ch
        ch_mult       = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout       = config.model.dropout
        in_channels   = config.model.in_channels
        # Change to support different width and height
        if hasattr(config.dataset, 'image_height') and hasattr(config.dataset, 'image_width'):
            self.height = config.dataset.image_height
            self.width = config.dataset.image_width
        else:
            self.height = config.dataset.image_size
            self.width = config.dataset.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_steps

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # --- Timestep embedding ---
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # --- Main input downsampling conv ---
        self.conv_in = nn.Conv2d(
            in_channels, self.ch,
            kernel_size=3, stride=1,
            padding=1, padding_mode='circular'
        )

        # --- y0 embedding ---
        self.y0_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.ch, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        )

        # NOTE: we have up to 3 embeddings: x->ch, dx->ch, y0->ch => total 3ch
        self.combine_conv = nn.Conv2d(
            in_channels=self.ch * 2,  # x_emb + y0_emb
            out_channels=self.ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # --- Downsampling ---
        curr_height = self.height
        curr_width = self.width
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]      # e.g. 64*(1) = 64 in first level
            block_out = ch * ch_mult[i_level]        # e.g. 64*(1) = 64

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                block_in = block_out
                curr_min_dim = min(curr_height, curr_width)
                if curr_min_dim in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down_level = nn.Module()
            down_level.block = block
            down_level.attn = attn

            if i_level != self.num_resolutions - 1:
                down_level.downsample = Downsample(block_in, resamp_with_conv)
                curr_height = curr_height // 2
                curr_width = curr_width // 2

            self.down.append(down_level)

        # --- Middle ---
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # --- Upsampling ---
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = ch * ch_mult[i_level]
            skip_in   = ch * ch_mult[i_level]

            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout
                    )
                )
                block_in = block_out
                curr_min_dim = min(curr_height, curr_width)
                if curr_min_dim in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up_level = nn.Module()
            up_level.block = block
            up_level.attn = attn

            if i_level != 0:
                up_level.upsample = Upsample(block_in, resamp_with_conv)
                curr_height = curr_height * 2
                curr_width = curr_width * 2

            self.up.insert(0, up_level)  # prepend to maintain correct order

        # --- Final norm + conv ---
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular'
        )

    ###################################
    # Forward
    ###################################
    def forward(self, x, t=None, y0=None):
        """
        :param x:   The main input (e.g. x_t) [N, C, H, W].
        :param t:   Timestep(s) for diffusion embedding.
        :param dx:  (Optional) PDE gradient / physics info, same shape as x.
        :param y0:  (Optional) degraded image for conditioning, same shape as x.
        :return:    The model output [N, out_ch, H, W].
        """
        # --- Timestep embedding ---
        temb = None
        if t is not None:
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)

        # --- Convolution of x ---
        x_main = self.conv_in(x)  # shape: [N, ch, H, W]

        # --- y0 embedding ---
        if y0 is not None:
            y0_emb = self.y0_conv(y0)
        else:
            y0_emb = torch.zeros_like(x_main)

        # --- Concatenate: x + dx + y0 => 3ch
        x_cat = torch.cat([x_main, y0_emb], dim=1)

        # --- Combine down to self.ch channels ---
        x_in = self.combine_conv(x_cat)  # shape: [N, ch, H, W]

        # --- Downsample pass ---
        hs = [x_in]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # --- Middle ---
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # --- Upsample pass ---
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # --- Final output ---
        h = self.norm_out(h)
        h = nonlinearity(h)
        out = self.conv_out(h)
        return out
