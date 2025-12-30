import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from torch.cuda.amp import autocast

from rfwave.mel_processor import MelProcessor
from rfwave.resample import ResampleModule
from einops import rearrange


@dataclass
class RFMelConfig:
    """Configuration for the RFWave diffusion model."""
    solver: str = "euler"
    noise_std: float = 1.0
    t_scheduler: str = "linear"
    training_cfg_rate: float = 0.0
    inference_cfg_rate: float = 3.0
    n_timesteps: int = 10
    batch_mul: int = 1
    patch_size: int = 8
    num_prefix_patches: int = 0


class RFMel(torch.nn.Module):
    def __init__(self,
                 estimator: torch.nn.Module,
                 mel_processor: MelProcessor,
                 resample_module: ResampleModule,
                 config: RFMelConfig,
                 use_skip_connection: bool = False,
                 ):
        super().__init__()
        self.config = config
        self.solver = config.solver
        self.noise_std = config.noise_std
        self.t_scheduler = config.t_scheduler
        self.training_cfg_rate = config.training_cfg_rate
        self.inference_cfg_rate = config.inference_cfg_rate
        self.num_prefix_patches = int(getattr(config, 'num_prefix_patches', 0))

        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.mel_processor = mel_processor
        self.resample_module = resample_module
        self.patch_size = self.config.patch_size

        if self.num_prefix_patches > 0:
            assert getattr(self.estimator, 'prev_cond', False), (
                "Estimator must be created with prev_cond=True when num_prefix_patches > 0"
            )
        
        # Skip connection from downsample output to upsample input
        self.use_skip_connection = use_skip_connection
        # Learnable scale for skip features to balance with mu
        self.skip_scale = torch.nn.Parameter(torch.ones(1)) if use_skip_connection else None
        
        # Learnable positional embedding for intra-patch positions
        # Shape: (1, upsample_output_dim, patch_size)
        # Use the actual output dimension from resample_module.upsample, not mel_processor config
        upsample_output_dim = resample_module.complex_spec_dim
        self.intra_patch_pe = torch.nn.Parameter(
            torch.zeros(1, upsample_output_dim, self.patch_size)
        )
        torch.nn.init.trunc_normal_(self.intra_patch_pe, std=0.02)

    @property
    def use_cfg(self):
        return self.training_cfg_rate > 0.

    @property
    def use_prefix(self):
        return self.num_prefix_patches > 0
    
    def apply_skip_connection(self, mu: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """Apply skip connection at input side (before upsample).
        
        Args:
            mu: LLM hidden states (main signal)
                shape: (batch_size, llm_hidden_dim, num_patches)
            skip_features: Downsample output features (residual details)
                shape: (batch_size, llm_hidden_dim, num_patches)
        
        Returns:
            Fused features in hidden space (before upsample)
                shape: (batch_size, llm_hidden_dim, num_patches)
        """
        # mu is the main signal, skip_features are scaled residual details
        return mu + self.skip_scale * skip_features

    def patchify(self, comp_spec):
        # No-overlap patchify (aligned with Wave)
        return rearrange(comp_spec, 'b c (n p) -> (b n) c p', p=self.patch_size)

    def unpatchify(self, patches, n_patches):
        # No-overlap unpatchify
        return rearrange(patches, '(b n) c p -> b c (n p)', n=n_patches)

    # Removed: upsample (no longer needed without overlap)

    def patchify_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        if self.num_prefix_patches <= 0:
            return prefix
        prefix_l = self.num_prefix_patches * self.patch_size
        prefix = F.pad(prefix, (prefix_l, 0))
        prefix = prefix.unfold(dimension=2, size=prefix_l, step=self.patch_size)
        prefix = rearrange(prefix, 'b c n p -> (b n) c p')
        return prefix

    def pad_for_prefix(self, mu: torch.Tensor, zt: torch.Tensor, prefix: torch.Tensor):
        prefix_l = self.num_prefix_patches * self.patch_size
        new_mu = F.pad(mu, (prefix_l, 0))
        new_zt = F.pad(zt, (prefix_l, 0))
        new_prefix = F.pad(prefix, (0, self.patch_size))
        return new_mu, new_zt, new_prefix

    def add_noise_to_prefix(self, prefix: torch.Tensor, min_t: float = 0.6) -> torch.Tensor:
        if not self.training:
            return prefix
        b, _, T_pref = prefix.shape
        # t ∈ [min_t, 1.0]: from min_t (more noise) to 1.0 (pure data)
        t = torch.rand((b,), device=prefix.device) * (1.0 - min_t) + min_t
        t = t.to(prefix.dtype).view(-1, 1, 1)
        e_pref = self.get_noise(b, T_pref, device=prefix.device).to(prefix.dtype)
        # Flow matching: t=0 is noise, t=1 is data
        return t * prefix + (1 - t) * e_pref

    def build_zero_prefix(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Build zero prefix for mel generation.
        
        Args:
            batch_size: Batch size
            device: Device
            dtype: Data type
        
        Returns:
            Zero prefix tensor, shape: (batch_size, n_mels, prefix_length)
        """
        c = self.estimator.output_channels
        l = self.num_prefix_patches * self.patch_size
        zero_prefix = torch.zeros(batch_size, c, l, device=device, dtype=dtype)
        return zero_prefix

    @torch.inference_mode()
    def sample(self, mu, n_timesteps=None, cfg_rate=None, temperature=1.0, one_patch=False, prefix=None, skip_features=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            one_patch: whether generating one patch at a time
            prefix: optional prefix for consistency
            skip_features (torch.Tensor, optional): downsample output from previous patch for skip connection
                shape: (batch_size, llm_hidden_dim, 1)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if n_timesteps is None:
            n_timesteps = self.config.n_timesteps
        if cfg_rate is None:
            cfg_rate = self.config.inference_cfg_rate

        b, n, c = mu.shape
        # assert n == 1, f"During inference, sample only support 1 patch, however {t} patches is provided."
        p = self.patch_size
        z = self.get_noise(b, n * p, mu.device, temperature=temperature).to(mu)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # t_span = torch.sin(t_span * math.pi / 2) # 加强noise侧的采样
        # NOTE: transpose mu from (B, N, C) to (B, C, N) to match downstream expectations
        mu = mu.mT
        return self.solve_euler(z, t_span=t_span, mu=mu, cfg_rate=cfg_rate, one_patch=one_patch, prefix=prefix, skip_features=skip_features)

    def solve_euler(self, z, t_span, mu, cfg_rate=0., one_patch=False, prefix=None, skip_features=None):
        """
        Fixed euler solver for ODEs.
        Args:
            z (torch.Tensor): random noise (initial state)
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            cfg_rate: classifier-free guidance rate
            one_patch: whether generating one patch at a time
            prefix: optional prefix for consistency
            skip_features: downsample output from previous patch
        """
        prefix_l = self.num_prefix_patches * self.patch_size
        if self.use_prefix:
            assert prefix is not None, "prefix is required when num_prefix_patches > 0"

        sol = []

        b = z.size(0)
        
        # Apply skip connection at input side if enabled
        if self.use_skip_connection and skip_features is not None:
            mu = self.apply_skip_connection(mu, skip_features)
        
        # Upsample (with or without skip features)
        mu = self.resample_module.upsample(mu)
        mu = self.patchify(mu)
        z = self.patchify(z)

        n_patches = mu.size(0) // b
        if one_patch:
            assert n_patches == 1, f"Only 1 patch is supported for sampling, however, {n_patches} patches is provided."
        else:
            assert not self.use_prefix, "Prefix is not supported for non-one-patch sampling."
        
        # Add intra-patch positional embedding to mu (after upsample and patchify)
        mu = mu + self.intra_patch_pe

        if self.use_prefix:
            mu, z, prefix = self.pad_for_prefix(mu, z, prefix)

        b = z.size(0)
        for step in range(len(t_span) - 1):
            t, t_next = t_span[step], t_span[step + 1]
            dt = t_next - t

            if self.use_cfg:
                n_channels = z.size(1)
                z_in = torch.zeros([2*b, n_channels, z.size(2)], device=z.device, dtype=z.dtype)
                mu_in = torch.zeros([2*b, mu.size(1), mu.size(2)], device=z.device, dtype=z.dtype)
                t_in = torch.zeros([2*b], device=z.device, dtype=z.dtype)
                z_in[:b], z_in[b:] = z, z
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                if self.use_prefix:
                    prefix_in = torch.zeros([2*b, prefix.size(1), prefix.size(2)], device=z.device, dtype=z.dtype)
                    prefix_in[:b] = prefix
                else:
                    prefix_in = None
                v_pred = self.estimator(z_in, t_in, mu_in, prev_cond=prefix_in)
                v, v_uncond = torch.split(v_pred, [b, b], dim=0)
                v = v_uncond + cfg_rate * (v - v_uncond)
            else:
                z_in = z
                mu_in = mu
                t_in = t.unsqueeze(0).expand(b)
                v = self.estimator(z_in, t_in, mu_in, prev_cond=prefix)

            # z_next = z + dt * v
            z[..., prefix_l:] = z[..., prefix_l:] + dt * v[..., prefix_l:]
            
            if one_patch:
                sol.append(z[..., prefix_l:])
            else:
                sol.append(self.unpatchify(z, n_patches))

        if one_patch:
            return sol[-1]  # return complex spectrum directly.
        else:
            return self.mel_processor.revert_norm_mel(sol[-1])

    def get_noise(self, batch_size, num_frames, device, temperature=1.):
        # get noise (e) in time domain and transform to spec domain
        padding = self.mel_processor.config.padding
        n_mels = self.mel_processor.config.n_mels

        nf = num_frames if padding == "same" else (num_frames - 1)
        r = torch.randn([batch_size, n_mels, nf], device=device) * temperature
        return r * self.noise_std

    def compute_loss(self, x, mu, prefix: torch.Tensor = None, skip_features: torch.Tensor = None):
        """Computes diffusion loss

        Args:
            x (torch.Tensor): Target mel spectrogram
                shape: (batch_size, n_mels, n_frames)
            mu (torch.Tensor): output of encoder (LLM hidden states)
                shape: (batch_size, llm_hidden_dim, num_patches)
            prefix: optional prefix for consistency
            skip_features (torch.Tensor, optional): downsample output for skip connection
                shape: (batch_size, llm_hidden_dim, num_patches)
        Returns:
            loss: conditional flow matching loss

        """
        # Apply skip connection at input side if enabled
        if self.use_skip_connection and skip_features is not None:
            mu = self.apply_skip_connection(mu, skip_features)
        
        # Upsample (with or without skip features)
        mu = self.resample_module.upsample(mu)
        prefix_l = self.num_prefix_patches * self.patch_size
        if self.use_prefix:
            assert prefix is not None, "prefix is required when num_prefix_patches > 0"

        if self.config.batch_mul > 1:
            x = torch.repeat_interleave(x, self.config.batch_mul, 0)
            mu = torch.repeat_interleave(mu, self.config.batch_mul, 0)
            if self.use_prefix:
                prefix = torch.repeat_interleave(prefix, self.config.batch_mul, 0)

        b, _, T = x.shape

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            if self.use_prefix:
                prefix = prefix * cfg_mask.view(-1, 1, 1)

        # random timestep
        t = torch.rand((mu.size(0),), device=mu.device).to(x.dtype)
        e = self.get_noise(b, T, device=mu.device).to(x.dtype)

        # patchify
        x = self.patchify(x)
        e = self.patchify(e)
        mu = self.patchify(mu)
        n_patches = mu.size(0) // t.size(0)
        t = torch.repeat_interleave(t, n_patches, dim=0)
        
        # Add intra-patch positional embedding to mu (after upsample and patchify)
        mu = mu + self.intra_patch_pe

        # get velocity: from noise (t=0) to data (t=1)
        z = (1 - t.view(-1, 1, 1)) * e + t.view(-1, 1, 1) * x
        v = x - e  # velocity points from e to x

        if self.use_prefix:
            prefix = self.add_noise_to_prefix(prefix)
            prefix = self.patchify_prefix(prefix)
            mu, z, prefix = self.pad_for_prefix(mu, z, prefix)

        pred = self.estimator(z, t, mu, prev_cond=prefix)

        if self.use_prefix:
            pred = pred[..., prefix_l:]
            z = z[..., prefix_l:]

        pred = self.unpatchify(pred, n_patches)
        tgt = self.unpatchify(v.detach(), n_patches)

        loss = F.mse_loss(pred, tgt)
        return loss
