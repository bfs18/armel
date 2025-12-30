# #####################################################################################
# ##                  Crafted with <3 by Peng Liu                                    ##
# ##                                                                                 ##
# ##                  - feanorliu@tencent.com                                        ##
# ##                  - laupeng1989@gmail.com                                        ##
# #####################################################################################

import typing as tp
import torch
import math
import torch.nn.functional as F
import logging
from torch import nn
from einops import rearrange, repeat
from rfwave.pqmf_equalizer import MeanStdProcessor
from rfwave.perceiver import PerceiverResampler

logger = logging.getLogger(__name__)


def _init_resample_weights(root: nn.Module):
    for name, module in root.named_modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


def calculate_padding_for_exact_length(input_length: int, target_length: int, kernel_size: int, stride: int) -> int:
    """Calculate padding needed to achieve exact target length with convolution."""
    # For downsampling: output_length = (input_length + 2*padding - kernel_size) // stride + 1
    # Solving for padding: padding = ((target_length - 1) * stride + kernel_size - input_length) / 2
    padding = ((target_length - 1) * stride + kernel_size - input_length) / 2
    return max(0, int(math.ceil(padding)))


def calculate_padding_for_exact_length_transpose(input_length: int, target_length: int, kernel_size: int, stride: int) -> int:
    """Calculate padding needed to achieve exact target length with transposed convolution."""
    # For upsampling: output_length = (input_length - 1) * stride - 2*padding + kernel_size
    # Solving for padding: padding = ((input_length - 1) * stride + kernel_size - target_length) / 2
    padding = ((input_length - 1) * stride + kernel_size - target_length) / 2
    return max(0, int(round(padding)))



class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    
    Args:
        stride: Downsampling factor
        in_channels: Number of input channels
        out_channels: Number of output channels (if None, same as in_channels)
        channel_wise: Whether to use channel-wise convolution (only works when in_channels == out_channels)
        padding_mode: Padding strategy - 'valid' (no padding), 'same' (preserve length), 'exact' (calculate for exact output length)
    """

    def __init__(
        self,
        stride: int,
        in_channels: tp.Optional[int] = None,
        out_channels: tp.Optional[int] = None,
        channel_wise: bool = False,
        padding_mode: str = 'valid',
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = 2 * stride
        self.padding_mode = padding_mode
        self.channel_wise = channel_wise
        groups = 1
        assert in_channels is not None, "in_channels required for learnt convolutions."
        if out_channels is None:
            out_channels = in_channels
        if channel_wise:
            assert in_channels == out_channels, "channel_wise requires in_channels == out_channels"
            groups = in_channels

        # Calculate padding based on mode
        if padding_mode == 'valid':
            padding = 0
        elif padding_mode == 'same':
            # For 'same' padding, we want output_length = input_length / stride
            # This is achieved with padding = (kernel_size - stride) / 2
            padding = (self.kernel_size - stride) // 2
        elif padding_mode == 'exact':
            # For exact mode, we'll calculate padding dynamically
            padding = 0
        else:
            raise ValueError(f"Unknown padding_mode: {padding_mode}")

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        if self.padding_mode == 'exact':
            # Calculate padding to achieve exact output length
            input_length = x.shape[-1]
            # For downsampling, we want output_length = input_length / stride
            target_length = input_length // self.stride
            padding = calculate_padding_for_exact_length(
                input_length, target_length, self.kernel_size, self.stride
            )
            if padding > 0:
                x = F.pad(x, (padding, padding), mode='constant', value=0)
            y = self.conv(x)
        else:
            y = self.conv(x)
        return y


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    
    Args:
        stride: Upsampling factor
        in_channels: Number of input channels
        out_channels: Number of output channels (if None, same as in_channels)
        channel_wise: Whether to use channel-wise convolution (only works when in_channels == out_channels)
        padding_mode: Padding strategy - 'valid' (no padding), 'same' (preserve length), 'exact' (calculate for exact output length)
    """

    def __init__(
        self,
        stride: int,
        in_channels: tp.Optional[int] = None,
        out_channels: tp.Optional[int] = None,
        channel_wise: bool = False,
        padding_mode: str = 'valid',
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = 2 * stride
        self.padding_mode = padding_mode
        self.channel_wise = channel_wise
        groups = 1
        assert in_channels is not None, "in_channels required for learnt convolutions."
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        if channel_wise:
            assert in_channels == out_channels, "channel_wise requires in_channels == out_channels"
            groups = in_channels

        # Calculate padding based on mode
        if padding_mode == 'valid':
            padding = 0
        elif padding_mode == 'same':
            # For 'same' padding, we want output_length = input_length * stride
            # This is achieved with padding = (kernel_size - stride) / 2
            padding = (self.kernel_size - stride) // 2
        elif padding_mode == 'exact':
            # For exact mode, we'll calculate padding dynamically
            padding = 0
        else:
            raise ValueError(f"Unknown padding_mode: {padding_mode}")

        self.convtr = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        if self.padding_mode == 'exact':
            # Calculate padding to achieve exact output length
            input_length = x.shape[-1]
            # For upsampling, we want output_length = input_length * stride
            target_length = input_length * self.stride
            padding = calculate_padding_for_exact_length_transpose(
                input_length, target_length, self.kernel_size, self.stride
            )
            
            # Use F.conv_transpose1d with dynamic padding
            y = F.conv_transpose1d(
                x,
                self.convtr.weight,
                bias=None,
                stride=self.stride,
                padding=padding,
                groups=self.convtr.groups
            )
        else:
            y = self.convtr(x)
        return y


class PerceiverDownsampleStack(nn.Module):
    """
    Complete Perceiver-based downsample stack using PerceiverResampler.
    Processes entire sequence at once, compressing patch_size inputs to 1 output.
    Input: (B, C, patch_size * num_patches) -> Output: (B, llm_hidden_dim, num_patches)
    
    Args:
        input_dim: Input dimension (complex_spec_dim)
        output_dim: Output dimension (llm_hidden_dim)
        patch_size: Size of input patch (will be compressed to 1 token)
        depth: Number of perceiver layers
        heads: Number of attention heads
        dim_head: Dimension per head
        ff_mult: Feedforward expansion ratio
        augment_comp_spec: Whether to augment complex spectrogram
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_size: int,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4.0,
        augment_comp_spec: bool = True,
    ):
        super().__init__()
        self.augment_comp_spec = augment_comp_spec
        self.input_dim = int(input_dim * 1.5) if self.augment_comp_spec else input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        
        logger.info(f"PerceiverDownsampleStack: augment_comp_spec={augment_comp_spec}, "
                    f"input_dim={input_dim} -> actual_input_dim={self.input_dim}, "
                    f"output_dim={output_dim}, patch_size={patch_size}, depth={depth}")
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, output_dim)
        
        # Use PerceiverResampler with num_latents=1 to compress patch_size -> 1
        self.perceiver = PerceiverResampler(
            dim=output_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=1,  # Output 1 token per patch
            ff_mult=ff_mult,
        )
    
    def augment_comp_spec_func(self, comp_spec):
        r, i = torch.chunk(comp_spec, 2, dim=1)
        mag = torch.sqrt((r ** 2 + i ** 2).clamp_min_(1e-7))
        return torch.cat([r / mag, i / mag, torch.log(mag)], dim=1)
    
    def forward(self, x: torch.Tensor, cond_embedding_id: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input: (B, C, patch_size * num_patches)
        Output: (B, output_dim, num_patches)
        """
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        assert num_frames % self.patch_size == 0, \
            f"Number of frames must be divisible by patch_size, got {num_frames} % {self.patch_size} != 0"
        
        num_patches = num_frames // self.patch_size
        
        # Augment complex spectrogram if needed
        if self.augment_comp_spec:
            x = self.augment_comp_spec_func(x)
        
        # Reshape: (B, C, patch_size * num_patches) -> (B * num_patches, patch_size, C)
        x = rearrange(x, 'b c (n p) -> (b n) p c', p=self.patch_size)
        
        # Input projection: (B*num_patches, patch_size, input_dim) -> (B*num_patches, patch_size, output_dim)
        x = self.input_proj(x)
        
        # Apply PerceiverResampler: (B*num_patches, patch_size, output_dim) -> (B*num_patches, 1, output_dim)
        out = self.perceiver(x)
        
        # Reshape back: (B*num_patches, 1, output_dim) -> (B, output_dim, num_patches)
        out = rearrange(out, '(b n) l d -> b d (n l)', b=batch_size, n=num_patches)
        return out


class PerceiverUpsampleStack(nn.Module):
    """
    Complete Perceiver-based upsample stack using PerceiverResampler.
    Processes entire sequence at once, expanding 1 input to patch_size outputs.
    Input: (B, llm_hidden_dim, num_patches) -> Output: (B, complex_spec_dim, patch_size * num_patches)
    
    Args:
        input_dim: Input dimension (llm_hidden_dim)
        output_dim: Output dimension (complex_spec_dim)
        patch_size: Size of output patch per token (1 input will expand to patch_size outputs)
        depth: Number of perceiver layers
        heads: Number of attention heads
        dim_head: Dimension per head
        ff_mult: Feedforward expansion ratio
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_size: int,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        
        logger.info(f"PerceiverUpsampleStack: input_dim={input_dim}, output_dim={output_dim}, "
                    f"patch_size={patch_size}, depth={depth}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # Use PerceiverResampler with num_latents=patch_size to expand 1 -> patch_size
        self.perceiver = PerceiverResampler(
            dim=input_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=patch_size,  # Output patch_size tokens per input
            ff_mult=ff_mult,
        )
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, cond_embedding_id: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input: (B, input_dim, num_patches)
        Output: (B, output_dim, num_patches * patch_size)
        """
        batch_size = x.shape[0]
        num_patches = x.shape[2]
        
        # Reshape: (B, input_dim, num_patches) -> (B * num_patches, 1, input_dim)
        x = rearrange(x, 'b d n -> (b n) 1 d')
        
        # Input projection: (B*num_patches, 1, input_dim) -> (B*num_patches, 1, input_dim)
        x = self.input_proj(x)
        
        # Apply PerceiverResampler: (B*num_patches, 1, input_dim) -> (B*num_patches, patch_size, input_dim)
        out = self.perceiver(x)
        
        # Output projection: (B*num_patches, patch_size, input_dim) -> (B*num_patches, patch_size, output_dim)
        out = self.output_proj(out)
        
        # Reshape back: (B*num_patches, patch_size, output_dim) -> (B, output_dim, num_patches * patch_size)
        out = rearrange(out, '(b n) p d -> b d (n p)', b=batch_size, n=num_patches)
        return out


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


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
        adanorm_num_embeddings: tp.Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None and adanorm_num_embeddings > 1
        if self.adanorm:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    @torch.compile
    def forward(self, x: torch.Tensor, cond_embedding_id: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
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


# Helper to run a stack of ConvNeXtV2 blocks (pre/post stacks)
@torch.compile
def _forward_convnext_layers(x: torch.Tensor, layers: nn.ModuleList, cond_embedding_id: tp.Optional[torch.Tensor]):
    for block in layers:
        x = block(x, cond_embedding_id)
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


class DownsampleStack(nn.Module):
    """
    Downsample stack with ConvNeXt blocks and resample convolutions.
    Each stride level contains: ConvNeXt -> ConvDownsample1d
    
    With dim_scale_per_stage > 1, hidden dimension grows after each downsample:
    - Level 0: base_dim -> ConvNeXt -> Downsample -> base_dim * dim_scale_per_stage
    - Level 1: base_dim * dim_scale_per_stage -> ConvNeXt -> Downsample -> base_dim * dim_scale_per_stage^2
    - ...
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tp.Union[int, tp.List[int]],
        strides: tp.List[int],
        patch_size: int,
        intermediate_ratios: tp.Union[float, tp.List[float]] = None,
        channel_wise: bool = False,
        padding_mode: str = 'exact',
        pre_convnext_layers: int = 0,
        augment_comp_spec: bool = True,
        dim_scale_per_stage: float = 1.0,  # Dimension scale factor per downsample stage (e.g., 2.0 for doubling)
    ):
        super().__init__()
        self.augment_comp_spec = augment_comp_spec
        self.input_dim = int(input_dim * 1.5) if self.augment_comp_spec else input_dim
        self.output_dim = output_dim
        
        self.strides = strides
        self.num_levels = len(strides)
        self.patch_size = patch_size
        self.dim_scale_per_stage = dim_scale_per_stage
        
        # Handle hidden_dims: include all layer dimensions (input to each level + final output)
        if isinstance(hidden_dims, int):
            base_dim = hidden_dims
            # hidden_dims[i] is input to level i, hidden_dims[-1] is final output
            self.hidden_dims = [int(base_dim * (dim_scale_per_stage ** i)) for i in range(self.num_levels + 1)]
        else:
            self.hidden_dims = hidden_dims
            assert len(hidden_dims) == self.num_levels + 1, "hidden_dims length must be num_levels + 1"
        
        logger.info(f"DownsampleStack: augment_comp_spec={augment_comp_spec}, input_dim={input_dim} -> actual_input_dim={self.input_dim}, "
                    f"dim_scale_per_stage={dim_scale_per_stage}, hidden_dims={self.hidden_dims}")
        
        # Handle intermediate_ratios: convert float to list if needed
        if intermediate_ratios is None:
            self.intermediate_ratios = [3.0] * max(1, self.num_levels)
        elif isinstance(intermediate_ratios, (int, float)):
            self.intermediate_ratios = [float(intermediate_ratios)] * max(1, self.num_levels)
        else:
            self.intermediate_ratios = intermediate_ratios
            assert len(intermediate_ratios) == max(1, self.num_levels), \
                f"intermediate_ratios length must be {max(1, self.num_levels)} (max(1, num_levels))"

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        # Optional pre ConvNeXtV2 blocks at input resolution
        self.pre_convnext_layers = nn.ModuleList()
        for _ in range(max(0, int(pre_convnext_layers))):
            self.pre_convnext_layers.append(
                ConvNeXtV2Block(
                    dim=self.hidden_dims[0],
                    intermediate_dim=int(self.hidden_dims[0] * self.intermediate_ratios[0]),
                )
            )
        
        # Downsample levels
        self.downsample_levels = nn.ModuleList()
        for i in range(self.num_levels):
            in_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i + 1]
            
            level = nn.ModuleDict({
                'convnext': ConvNeXtV2Block(
                    dim=in_dim,
                    intermediate_dim=int(in_dim * self.intermediate_ratios[i]),
                ),
                'downsample': ConvDownsample1d(
                    stride=strides[i],
                    in_channels=in_dim,
                    out_channels=out_dim,
                    channel_wise=channel_wise if in_dim == out_dim else False,
                    padding_mode=padding_mode,
                )
            })
            self.downsample_levels.append(level)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dims[-1], output_dim)

        _init_resample_weights(self)

    def augment_comp_spec_func(self, comp_spec):
        r, i = torch.chunk(comp_spec, 2, dim=1)
        mag = torch.sqrt((r ** 2 + i ** 2).clamp_min_(1e-7))
        return torch.cat([r / mag, i / mag, torch.log(mag)], dim=1)

    def forward(self, x: torch.Tensor, cond_embedding_id: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        num_frames = x.shape[2]
        assert num_frames % self.patch_size == 0, f"Number of frames must be divisible by patch_size, got {num_frames} % {self.patch_size} != 0"

        if self.augment_comp_spec:
            x = self.augment_comp_spec_func(x)
        x = rearrange(x, 'b c (n p) -> (b n) c p', p=self.patch_size)

        # Input is (B, C, T), transpose for linear: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # Input projection: (B, T, C) -> (B, T, hidden_dims[0])
        x = self.input_proj(x)
        
        # Transpose for conv1d: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # Pre ConvNeXtV2 blocks at original resolution
        if len(self.pre_convnext_layers) > 0:
            x = _forward_convnext_layers(x, self.pre_convnext_layers, cond_embedding_id)
        
        # Downsample levels
        for i, level in enumerate(self.downsample_levels):
            # ConvNeXt block
            x = level['convnext'](x, cond_embedding_id)
            
            # Downsample
            x = level['downsample'](x)

        # Transpose back: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # Output projection: (B, T, hidden_dims[-1]) -> (B, T, output_dim)
        x = self.output_proj(x)
        
        # Transpose back to (B, C, T)
        x = x.transpose(1, 2)
        
        x = rearrange(x, '(b n) c p -> b c (n p)', n=num_frames // self.patch_size)
        return x


class UpsampleStack(nn.Module):
    """
    Upsample stack with ConvNeXt blocks and resample convolutions.
    Each stride level contains: ConvTrUpsample1d -> ConvNeXt
    
    With dim_scale_per_stage > 1, hidden dimension shrinks after each upsample:
    - Level 0: initial_dim -> Upsample -> initial_dim / dim_scale_per_stage -> ConvNeXt
    - Level 1: initial_dim / dim_scale_per_stage -> Upsample -> initial_dim / dim_scale_per_stage^2 -> ConvNeXt
    - ...
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tp.Union[int, tp.List[int]],
        strides: tp.List[int],
        patch_size: int,
        intermediate_ratios: tp.Union[float, tp.List[float]] = None,
        channel_wise: bool = False,
        padding_mode: str = 'exact',
        post_convnext_layers: int = 0,
        dim_scale_per_stage: float = 1.0,  # Dimension scale factor per upsample stage (e.g., 2.0 for halving)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.strides = strides
        self.num_levels = len(strides)
        self.patch_size = patch_size
        self.dim_scale_per_stage = dim_scale_per_stage
        
        # Handle hidden_dims: include all layer dimensions (initial input + output of each level)
        if isinstance(hidden_dims, int):
            base_dim = hidden_dims
            # hidden_dims[0] is initial input, hidden_dims[i+1] is output of level i
            self.hidden_dims = [int(base_dim * (dim_scale_per_stage ** (self.num_levels - i))) for i in range(self.num_levels + 1)]
        else:
            self.hidden_dims = hidden_dims
            assert len(hidden_dims) == self.num_levels + 1, "hidden_dims length must be num_levels + 1"
        
        logger.info(f"UpsampleStack: input_dim={input_dim}, output_dim={output_dim}, "
                    f"dim_scale_per_stage={dim_scale_per_stage}, hidden_dims={self.hidden_dims}")
        
        # Handle intermediate_ratios: convert float to list if needed
        if intermediate_ratios is None:
            self.intermediate_ratios = [3.0] * max(1, self.num_levels)
        elif isinstance(intermediate_ratios, (int, float)):
            self.intermediate_ratios = [float(intermediate_ratios)] * max(1, self.num_levels)
        else:
            self.intermediate_ratios = intermediate_ratios
            assert len(intermediate_ratios) == max(1, self.num_levels), \
                f"intermediate_ratios length must be {max(1, self.num_levels)} (max(1, num_levels))"
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, self.hidden_dims[0])
        
        # Upsample levels
        self.upsample_levels = nn.ModuleList()
        for i in range(self.num_levels):
            in_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i + 1]
            
            level = nn.ModuleDict({
                'upsample': ConvTrUpsample1d(
                    stride=strides[i],
                    in_channels=in_dim,
                    out_channels=out_dim,
                    channel_wise=channel_wise if in_dim == out_dim else False,
                    padding_mode=padding_mode,
                ),
                'convnext': ConvNeXtV2Block(
                    dim=out_dim,
                    intermediate_dim=int(out_dim * self.intermediate_ratios[i]),
                ),
            })
            self.upsample_levels.append(level)
        
        # Optional post ConvNeXtV2 blocks at final resolution
        self.post_convnext_layers = nn.ModuleList()
        for _ in range(max(0, int(post_convnext_layers))):
            self.post_convnext_layers.append(
                ConvNeXtV2Block(
                    dim=self.hidden_dims[-1],
                    intermediate_dim=int(self.hidden_dims[-1] * self.intermediate_ratios[-1]),
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dims[-1], output_dim)

        _init_resample_weights(self)

    def forward(self, x: torch.Tensor, cond_embedding_id: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        num_patches = x.shape[2]

        x = rearrange(x, 'b c (n p) -> (b n) c p', p=1)

        # Input is (B, C, T), transpose for linear: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # Input projection: (B, T, C) -> (B, T, hidden_dims[0])
        x = self.input_proj(x)
        
        # Transpose for conv1d: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # Upsample levels
        for i, level in enumerate(self.upsample_levels):
            # Upsample
            x = level['upsample'](x)

            # ConvNeXt block
            x = level['convnext'](x, cond_embedding_id)
        
        # Post ConvNeXtV2 blocks at final resolution
        if len(self.post_convnext_layers) > 0:
            x = _forward_convnext_layers(x, self.post_convnext_layers, cond_embedding_id)
        
        # Transpose back: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        
        # Output projection: (B, T, hidden_dims[-1]) -> (B, T, output_dim)
        x = self.output_proj(x)
        
        # Transpose back to (B, C, T)
        x = x.transpose(1, 2)
        x = rearrange(x, '(b n) c p -> b c (n p)', n=num_patches)
        
        return x


class ResampleModule(nn.Module):
    """
    Complete encoder-decoder with downsample and upsample stacks.
    Supports Conv-based and Perceiver Stack-based resampling.
    
    Args:
        complex_spec_dim: Input complex spectrogram dimension
        llm_hidden_dim: LLM hidden dimension (output dimension)
        patch_size: Patch size for resampling
        resample_type: 'conv' or 'perceiver_stack'
        
        # Conv-specific parameters (ignored if resample_type='perceiver_stack')
        hidden_dims: Hidden dimensions for Conv layers (base dimension if dim_scale_per_stage > 1)
        downsample_strides: Stride list for downsampling [2, 4] means 2x then 4x
        upsample_strides: Stride list for upsampling (default: reversed downsample_strides)
        intermediate_ratios: Expansion ratio for intermediate layers
        channel_wise: Whether to use channel-wise convolution
        padding_mode: Padding mode ('exact', 'same', 'valid')
        io_convnext_layers: Number of ConvNeXt layers at input/output
        dim_scale_per_stage: Dimension scale factor per stage (e.g., 2.0 for doubling/halving)
        
        # Perceiver-specific parameters (ignored if resample_type='conv')
        perceiver_depth: Number of Perceiver transformer layers
        perceiver_heads: Number of attention heads
        
        # Common parameters
        use_io_norm: Whether to use LayerNorm at input/output
        augment_comp_spec: Whether to augment complex spectrogram (split magnitude/phase)
    """
    
    def __init__(
        self,
        complex_spec_dim: int,
        llm_hidden_dim: int,
        patch_size: int,
        resample_type: str = 'conv',
        # Conv-specific parameters (ignored if resample_type='perceiver_stack')
        hidden_dims: tp.Union[int, tp.List[int]] = 512,
        downsample_strides: tp.List[int] = None,
        upsample_strides: tp.List[int] = None,
        intermediate_ratios: tp.Union[float, tp.List[float]] = None,
        channel_wise: bool = False,
        padding_mode: str = 'exact',
        io_convnext_layers: int = 0,
        dim_scale_per_stage: float = 2.0,
        # Perceiver-specific parameters (ignored if resample_type='conv')
        perceiver_depth: int = 6,
        perceiver_heads: int = 8,
        # Common parameters
        use_io_norm: bool = False,
        augment_comp_spec: bool = False,
    ):
        super().__init__()
        
        # Set defaults for strides
        if downsample_strides is None:
            downsample_strides = []
        if upsample_strides is None:
            upsample_strides = list(reversed(downsample_strides))
        
        # Validate: patch_size == prod(downsample_strides)
        stride_product = 1
        for s in downsample_strides:
            stride_product *= s
        if stride_product != patch_size:
            raise ValueError(
                f"patch_size ({patch_size}) must equal product of downsample_strides "
                f"({downsample_strides} = {stride_product})"
            )
        
        # Prepare parameters for upsampler (reverse the structure for conv mode)
        if isinstance(hidden_dims, int):
            upsample_hidden_dims = hidden_dims
        else:
            upsample_hidden_dims = list(reversed(hidden_dims))
        
        if isinstance(intermediate_ratios, (int, float)):
            upsample_intermediate_ratios = intermediate_ratios
        elif intermediate_ratios is None:
            upsample_intermediate_ratios = None
        else:
            upsample_intermediate_ratios = list(reversed(intermediate_ratios))

        # Choose between Conv-based and Perceiver Stack-based
        if resample_type == 'conv':
            self.downsampler = DownsampleStack(
                input_dim=complex_spec_dim,
                output_dim=llm_hidden_dim,
                hidden_dims=hidden_dims,
                strides=downsample_strides,
                patch_size=patch_size,
                intermediate_ratios=intermediate_ratios,
                channel_wise=channel_wise,
                padding_mode=padding_mode,
                pre_convnext_layers=io_convnext_layers,
                augment_comp_spec=augment_comp_spec,
                dim_scale_per_stage=dim_scale_per_stage,
            )
            
            self.upsampler = UpsampleStack(
                input_dim=llm_hidden_dim,
                output_dim=complex_spec_dim,
                hidden_dims=upsample_hidden_dims,
                strides=upsample_strides,
                patch_size=patch_size,
                intermediate_ratios=upsample_intermediate_ratios,
                channel_wise=channel_wise,
                padding_mode=padding_mode,
                post_convnext_layers=io_convnext_layers,
                dim_scale_per_stage=dim_scale_per_stage,
            )
        elif resample_type == 'perceiver':
            # Use complete Perceiver-based stacks
            self.downsampler = PerceiverDownsampleStack(
                input_dim=complex_spec_dim,
                output_dim=llm_hidden_dim,
                patch_size=patch_size,
                depth=perceiver_depth,
                heads=perceiver_heads,
                augment_comp_spec=augment_comp_spec,
            )
            
            self.upsampler = PerceiverUpsampleStack(
                input_dim=llm_hidden_dim,
                output_dim=complex_spec_dim,
                patch_size=patch_size,
                depth=perceiver_depth,
                heads=perceiver_heads,
            )
        else:
            raise ValueError(f"Unknown resample_type: {resample_type}. Choose 'conv' or 'perceiver'.")

        self.patch_size = patch_size
        self.complex_spec_dim = complex_spec_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.resample_type = resample_type
        
        logger.info(f"ResampleModule: resample_type={resample_type}, augment_comp_spec={augment_comp_spec}, "
                    f"complex_spec_dim={complex_spec_dim}, llm_hidden_dim={llm_hidden_dim}, "
                    f"patch_size={patch_size}, use_io_norm={use_io_norm}")
        
        # Add normalization layers for Transformer interface
        self.use_io_norm = use_io_norm

        if use_io_norm:
            self.pre_transformer_norm = nn.LayerNorm(llm_hidden_dim)
            self.post_transformer_norm = nn.LayerNorm(llm_hidden_dim)

    def downsample(self, complex_spec: torch.Tensor):
        assert complex_spec.shape[1] == self.complex_spec_dim, f"Complex spec shape must be (B, {self.complex_spec_dim}, {self.patch_size})"
        dtype = complex_spec.dtype
        out = self.downsampler(complex_spec)
        
        if self.use_io_norm:
            # (B, C, T) -> (B, T, C) for LayerNorm
            out = out.transpose(1, 2)
            out = self.pre_transformer_norm(out.float())
            out = out.transpose(1, 2)
        
        return out.to(dtype)

    def upsample(self, encoded: torch.Tensor):
        assert encoded.shape[1] == self.llm_hidden_dim, f"Encoded shape must be (B, {self.llm_hidden_dim}, 1)"
        dtype = encoded.dtype

        if self.use_io_norm:
            # (B, C, T) -> (B, T, C) for LayerNorm
            encoded = encoded.transpose(1, 2)
            encoded = self.post_transformer_norm(encoded.float())
            encoded = encoded.transpose(1, 2)
        
        out = self.upsampler(encoded.to(dtype))
        return out


def test_resample_stacks():
    """Test DownsampleStack and UpsampleStack with 8->4->1 downsampling and 1->4->8 upsampling."""
    print("=== Testing Resample Stacks ===")
    
    try:
        # Test DownsampleStack: 8 -> 4 -> 1
        print("--- Testing DownsampleStack (8->4->1) ---")
        downsample_stack = DownsampleStack(
            input_dim=64,
            output_dim=256,
            hidden_dims=128,
            strides=[2, 4],
            patch_size=8,
            intermediate_ratios=4.0,
            padding_mode='exact'
        )
        
        x = torch.randn(2, 64, 8)  # (batch, features, time)
        print(f"Input shape: {x.shape}")
        
        encoded = downsample_stack(x)
        print(f"DownsampleStack output shape: {encoded.shape}")
        assert encoded.shape == (2, 256, 1), f"Expected (2, 256, 1), got {encoded.shape}"
        
        # Test UpsampleStack: 1 -> 4 -> 8
        print("\n--- Testing UpsampleStack (1->4->8) ---")
        upsample_stack = UpsampleStack(
            input_dim=256,
            output_dim=64,
            hidden_dims=128,
            strides=[4, 2],
            patch_size=8,
            intermediate_ratios=4.0,
            padding_mode='exact'
        )
        
        decoded = upsample_stack(encoded)
        print(f"UpsampleStack output shape: {decoded.shape}")
        assert decoded.shape == (2, 64, 8), f"Expected (2, 64, 8), got {decoded.shape}"
        
        # Test ResampleModule
        print("\n--- Testing ResampleModule ---")
        resample_module = ResampleModule(
            complex_spec_dim=64,
            llm_hidden_dim=256,
            patch_size=8,
            resample_type='conv',
            hidden_dims=128,
            downsample_strides=[2, 4],
            intermediate_ratios=4.0,
            padding_mode='exact'
        )
        
        # Test with (B, C, T) format
        complex_spec = torch.randn(2, 64, 8)  # (batch, complex_spec_dim, patch_size)
        encoded = resample_module.downsample(complex_spec)
        decoded = resample_module.upsample(encoded)
        
        print(f"Input: {complex_spec.shape}")
        print(f"Encoded: {encoded.shape}")
        print(f"Decoded: {decoded.shape}")
        assert complex_spec.shape == decoded.shape, f"Shape mismatch: {complex_spec.shape} != {decoded.shape}"
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()



def test_resample_stacks_seq():
    try:
        # Test ResampleModule
        print("\n--- Testing ResampleModule ---")
        resample_module = ResampleModule(
            complex_spec_dim=64,
            llm_hidden_dim=256,
            patch_size=8,
            resample_type='conv',
            hidden_dims=128,
            downsample_strides=[2, 4],
            intermediate_ratios=4.0,
            padding_mode='exact'
        )

        # Test with (B, C, T) format
        complex_spec = torch.randn(2, 64, 8 * 11)  # (batch, complex_spec_dim, patch_size)
        encoded = resample_module.downsample(complex_spec)
        decoded = resample_module.upsample(encoded)

        print(f"Input: {complex_spec.shape}")
        print(f"Encoded: {encoded.shape}")
        print(f"Decoded: {decoded.shape}")
        assert complex_spec.shape == decoded.shape, f"Shape mismatch: {complex_spec.shape} != {decoded.shape}"

        print("✓ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_perceiver_stack():
    """Test PerceiverDownsampleStack and PerceiverUpsampleStack"""
    print("\n=== Testing Perceiver Stacks ===")
    
    try:
        # Test PerceiverDownsampleStack
        print("--- Testing PerceiverDownsampleStack ---")
        downsample_stack = PerceiverDownsampleStack(
            input_dim=64,
            output_dim=256,
            patch_size=8,
            depth=4,
            heads=8,
            augment_comp_spec=False,
        )
        
        x = torch.randn(2, 64, 8 * 3)  # (batch, complex_spec_dim, patch_size * num_patches)
        print(f"Input shape: {x.shape}")
        
        encoded = downsample_stack(x)
        print(f"PerceiverDownsampleStack output shape: {encoded.shape}")
        assert encoded.shape == (2, 256, 3), f"Expected (2, 256, 3), got {encoded.shape}"
        
        # Test PerceiverUpsampleStack
        print("\n--- Testing PerceiverUpsampleStack ---")
        upsample_stack = PerceiverUpsampleStack(
            input_dim=256,
            output_dim=64,
            patch_size=8,
            depth=4,
            heads=8,
        )
        
        decoded = upsample_stack(encoded)
        print(f"PerceiverUpsampleStack output shape: {decoded.shape}")
        assert decoded.shape == (2, 64, 8 * 3), f"Expected (2, 64, 24), got {decoded.shape}"
        
        # Test ResampleModule with perceiver_stack
        print("\n--- Testing ResampleModule with perceiver_stack ---")
        resample_module = ResampleModule(
            complex_spec_dim=64,
            llm_hidden_dim=256,
            patch_size=8,
            resample_type='perceiver_stack',
            perceiver_depth=4,
            perceiver_heads=8,
            augment_comp_spec=False,
        )
        
        # Test with (B, C, T) format
        complex_spec = torch.randn(2, 64, 8 * 5)  # (batch, complex_spec_dim, patch_size * num_patches)
        encoded = resample_module.downsample(complex_spec)
        decoded = resample_module.upsample(encoded)
        
        print(f"Input: {complex_spec.shape}")
        print(f"Encoded: {encoded.shape}")
        print(f"Decoded: {decoded.shape}")
        assert complex_spec.shape == decoded.shape, f"Shape mismatch: {complex_spec.shape} != {decoded.shape}"
        
        print("✓ All perceiver stack tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_all_resample_types():
    """Test both resample types: conv, perceiver_stack"""
    print("\n=== Testing All Resample Types ===")
    
    batch_size = 2
    complex_spec_dim = 64
    llm_hidden_dim = 256
    patch_size = 8
    num_patches = 5
    
    input_tensor = torch.randn(batch_size, complex_spec_dim, patch_size * num_patches)
    
    resample_types = ['conv', 'perceiver_stack']
    
    for resample_type in resample_types:
        try:
            print(f"\n--- Testing resample_type='{resample_type}' ---")
            
            resample_module = ResampleModule(
                complex_spec_dim=complex_spec_dim,
                llm_hidden_dim=llm_hidden_dim,
                patch_size=patch_size,
                resample_type=resample_type,
                hidden_dims=128,
                downsample_strides=[2, 4],
                perceiver_depth=4,
                perceiver_heads=8,
                augment_comp_spec=False,
            )
            
            encoded = resample_module.downsample(input_tensor)
            decoded = resample_module.upsample(encoded)
            
            print(f"  Input: {input_tensor.shape}")
            print(f"  Encoded: {encoded.shape}")
            print(f"  Decoded: {decoded.shape}")
            
            assert input_tensor.shape == decoded.shape, \
                f"Shape mismatch for {resample_type}: {input_tensor.shape} != {decoded.shape}"
            
            print(f"  ✓ {resample_type} passed!")
            
        except Exception as e:
            print(f"  ❌ {resample_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== All Resample Type Tests Complete ===")


if __name__ == "__main__":
    test_resample_stacks()
    test_resample_stacks_seq()
    test_perceiver_stack()
    test_all_resample_types()
