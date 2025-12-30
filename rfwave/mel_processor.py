"""
Mel Processor for AR-DiT-Mel project.

This module contains MelSpectrogramFeatures (inlined from rfwave_resample_model.py)
and MelProcessor for mel spectrogram extraction and normalization.
"""
from dataclasses import dataclass

import torch
import torchaudio
from torch import nn
from rfwave.pqmf_equalizer import MeanStdProcessor


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    return torch.log(torch.clip(x, min=clip_val))


class MelSpectrogramFeatures(nn.Module):
    """Extract mel spectrogram features from audio waveform.
    
    Inlined from rfwave_resample_model.py for isolation.
    """
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="same"):
        super().__init__()
        self.dim = n_mels
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        dtype = audio.dtype
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        # mel_spec must run in float32:
        # 1. cuFFT doesn't support bf16
        # 2. mel_scale's fb buffer needs matching dtype with input
        mel = self.mel_spec.float()(audio.float())
        features = safe_log(mel)
        return features.to(dtype)


@dataclass
class MelConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    padding: str = "same"


def create_mel_processor_from_config(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    padding: str = 'same',
) -> 'MelProcessor':
    """Helper function to create MelProcessor with consistent configuration."""
    mel_config = MelConfig(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        padding=padding,
    )
    return MelProcessor(mel_config)


class MelProcessor(nn.Module):
    def __init__(self, mel_config: MelConfig):
        super().__init__()
        self.config = mel_config
        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=mel_config.sample_rate,
            n_fft=mel_config.n_fft,
            hop_length=mel_config.hop_length,
            n_mels=mel_config.n_mels,
            padding=mel_config.padding)
        self.normalizer = MeanStdProcessor(dim=mel_config.n_mels)

    def get_norm_mel(self, audio):
        mel = self.mel_extractor(audio)
        norm_mel = self.normalizer.project_sample(mel)
        return norm_mel

    def get_norm_mel_from_linear(self, linear_spec: torch.Tensor) -> torch.Tensor:
        """Compute normalized Mel spectrogram from Linear Complex/Magnitude Spectrogram."""
        if linear_spec.size(1) % 2 == 0:
            real, imag = torch.chunk(linear_spec, 2, dim=1)
            mag_spec = torch.sqrt(real.float()**2 + imag.float()**2 + 1e-9)
        else:
            mag_spec = linear_spec.float()
            
        mel_scale_fn = self.mel_extractor.mel_spec.mel_scale
        mel = mel_scale_fn(mag_spec)
        log_mel = safe_log(mel)
        norm_mel = self.normalizer.project_sample(log_mel)
        return norm_mel.to(linear_spec.dtype)

    def revert_norm_mel(self, norm_mel):
        mel = self.normalizer.return_sample(norm_mel)
        return mel
