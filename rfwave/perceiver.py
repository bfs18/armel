import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from functools import wraps


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)


def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        # self.scale is no longer needed, as it's handled by the built-in function
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        # NOTE: Removed output norm - normalization is handled by ResampleModule's use_io_norm
        # self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

        for attn, ff in self.layers:
            nn.init.zeros_(attn.to_out.weight)
            nn.init.zeros_(ff[-1].weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.compile
    def forward(self, x):
        assert x.ndim == 3, f'Input tensor must be 3D (batch, seq, dim), but got {x.ndim}D'

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # NOTE: Return without normalization - handled by ResampleModule
        # return self.norm(latents)
        return latents


if __name__ == '__main__':
    model = PerceiverResampler(
        dim=512,
        depth=2,
        heads=8,
        num_latents=64
    )

    mock_input = torch.randn(4, 177, 512)
    output = model(mock_input)

    print(f"Input tensor shape: {mock_input.shape}")
    print(f"Output tensor shape: {output.shape}")

    expected_shape = (mock_input.shape[0], 64, 512)
    assert output.shape == expected_shape, f"Output shape mismatch! Expected: {expected_shape}, Got: {output.shape}"

    print("\nCode with F.scaled_dot_product_attention verified successfully!")
