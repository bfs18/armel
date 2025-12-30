import math
import torch

from torch import nn
from typing import Union, Optional, List


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim, groups):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.groups = groups

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        if self.groups == 1:
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        else:
            Gx = Gx.view(*Gx.shape[:2], self.groups, -1)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            Nx = Nx.view(*Nx.shape[:2], -1)
        return self.gamma * (x * Nx) + self.beta + x


class GroupLayerNorm(nn.Module):
    def __init__(self, groups: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.groups = groups
        self.scale = nn.Parameter(torch.ones([groups, embedding_dim // groups]))
        self.shift = nn.Parameter(torch.zeros([groups, embedding_dim // groups]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape[:-1]
        x = x.reshape(*sh, self.groups, -1)
        x = nn.functional.layer_norm(x, (self.dim // self.groups,), eps=self.eps)
        x = x * self.scale + self.shift
        return x.reshape(*sh, -1)


class GroupLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 groups: int = 1, device=None, dtype=None) -> None:
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        super().__init__(in_features // groups, out_features, bias, device, dtype)

    def forward(self, input):
        if self.groups == 1:
            return super().forward(input)
        else:
            sh = input.shape[:-1]
            input = input.view(*sh, self.groups, -1)
            weight = self.weight.view(self.groups, -1, self.weight.shape[-1])
            output = torch.einsum('...gi,...goi->...go', input, weight)
            output = output.reshape(*sh, -1) + self.bias
            return output


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        adanorm_num_embeddings: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None and adanorm_num_embeddings > 1
        if self.adanorm:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        elif groups == 1:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        else:
            self.norm = GroupLayerNorm(groups, dim, eps=1e-6)
        self.pwconv1 = GroupLinear(dim, intermediate_dim, groups=groups)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim, groups=groups)
        self.pwconv2 = GroupLinear(intermediate_dim, dim, groups=groups)

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(x.dtype)


class Base2FourierFeatures(nn.Module):
    def __init__(self, start=0, stop=8, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, inputs):
        freqs = range(self.start, self.stop, self.step)

        # Create Base 2 Fourier features
        w = 2. ** (torch.tensor(freqs, dtype=inputs.dtype)).to(inputs.device) * 2 * torch.pi
        w = torch.tile(w[None, :, None], (1, inputs.shape[1], 1))

        # Compute features
        h = torch.repeat_interleave(inputs, len(freqs), dim=1)
        h = w * h
        h = torch.stack([torch.sin(h), torch.cos(h)], dim=2)
        return h.reshape(h.size(0), -1, h.size(3))


class RFBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        num_bands: Optional[int] = None,
        dilation: Union[int, List[int]] = 1,
        prev_cond: Optional[bool] = False,
        pe_scale: float = 1000.,
        with_fourier_features: bool = False,
        use_dt: bool = False,
    ):
        super().__init__()
        self.prev_cond = prev_cond
        self.output_channels = output_channels
        self.with_fourier_features = with_fourier_features
        self.num_bands = num_bands
        self.use_dt = use_dt
        if self.with_fourier_features:
            self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
            fourier_dim = output_channels * 2 * (
                    (self.fourier_module.stop - self.fourier_module.start) // self.fourier_module.step)
        else:
            fourier_dim = 0
        mel_ch = input_channels
        input_channels = mel_ch + output_channels if prev_cond else mel_ch
        self.input_channels = mel_ch
        self.embed = nn.Conv1d(input_channels + output_channels + fourier_dim, dim, kernel_size=7, padding=3)
        self.adanorm = num_bands is not None and num_bands > 1
        if self.adanorm:
            self.norm = AdaLayerNorm(num_bands, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        if isinstance(dilation, (list, tuple)):
            assert num_layers % len(dilation) == 0, "num_layers must be divisible by len(dilation) for cycled dilation"
            dilation_cycles = dilation * (num_layers // len(dilation))
        else:
            assert dilation is None or isinstance(dilation, int), "dilation must be an int or a list of ints"
            dilation_cycles = [dilation] * num_layers  # None also in this case.
        self.convnext = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    adanorm_num_embeddings=num_bands,
                    dilation=dilation_cycles[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pe_scale = pe_scale
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), nn.GELU(), torch.nn.Linear(dim * 4, dim))
        if self.use_dt:
            self.dt_mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 4), nn.GELU(), torch.nn.Linear(dim * 4, dim))
        else:
            self.dt_mlp = None
        self.out = nn.Linear(dim, output_channels)
        self.apply(self._init_weights)
        nn.init.constant_(self.out.weight, 0.)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.time_mlp[0].weight)
        if self.use_dt:
            nn.init.xavier_uniform_(self.dt_mlp[0].weight)

    @staticmethod
    def get_out(out_layer, x):
        x = out_layer(x).transpose(1, 2)
        return x

    @torch.compile
    def forward(self, z_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor, prev_cond: Optional[torch.Tensor] = None,
                bandwidth_id=None,  dt_base: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prev_cond:
            assert prev_cond is not None, "prev_cond is required when prev_cond is True"
            x = torch.cat([x, prev_cond], dim=1)

        if self.with_fourier_features:
            z_t_f = self.fourier_module(z_t)
            x = self.embed(torch.cat([z_t, x, z_t_f], dim=1))
        else:
            x = self.embed(torch.cat([z_t, x], dim=1))
        emb_t = self.time_mlp(self.time_pos_emb(t, scale=self.pe_scale)).unsqueeze(2)

        if self.use_dt:
            assert dt_base is not None
            emb_dt = self.dt_mlp(self.time_pos_emb(dt_base, scale=self.pe_scale)).unsqueeze(2)
        else:
            assert dt_base is None
            emb_dt = 0.

        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x + emb_t + emb_dt, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.get_out(self.out, x)
        return x
