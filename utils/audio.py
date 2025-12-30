import base64
import pydub
import io
import numpy as np


def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        # 读取文件内容
        audio_bytes = audio_file.read()
        # 进行 Base64 编码
        base64_encoded_audio = base64.b64encode(audio_bytes)
    return base64_encoded_audio



def read_audio_from_base64(base64_encoded_audio, target_sr=None, audio_path=None):
    decoded_audio_bytes = base64.b64decode(base64_encoded_audio)
    audio_file_like_object = io.BytesIO(decoded_audio_bytes)
    file_format = audio_path.split('.')[-1] if audio_path else None
    audio = pydub.AudioSegment.from_file(audio_file_like_object, format=file_format)
    audio = audio.set_channels(1)
    if target_sr is not None:
        audio = audio.set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]

    max_val = 2 ** (audio.sample_width * 8 - 1)
    samples /= max_val
    audio_sr = audio.frame_rate
    return samples, audio_sr


def read_audio(audio_path, target_sr):
    audio = pydub.AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    if target_sr is not None:
        audio = audio.set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]

    max_val = 2 ** (audio.sample_width * 8 - 1)
    samples /= max_val
    audio_sr = audio.frame_rate
    return samples, audio_sr


import torch
from typing import List, Tuple, Union


def concat_waveforms_with_offsets(
    waveform_list: List[Union[np.ndarray, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate multiple waveforms and compute their start positions.
    
    Args:
        waveform_list: List of waveforms, each with shape (channels, length).
                      Can be numpy arrays or torch tensors.
    
    Returns:
        waveform_concat: Concatenated waveforms with shape (channels, total_length)
        waveform_start: Start index of each waveform in the concatenated tensor,
                       with shape (num_waveforms,)
    
    Example:
        >>> waveforms = [np.random.randn(1, 100), np.random.randn(1, 200)]
        >>> concat, offsets = concat_waveforms_with_offsets(waveforms)
        >>> concat.shape  # (1, 300)
        >>> offsets  # tensor([0, 100])
    """
    if not waveform_list:
        # Handle empty list case
        return torch.empty((1, 0), dtype=torch.float32), torch.tensor([0], dtype=torch.long)
    
    # Convert to torch tensors if needed
    waveform_tensors = [
        torch.from_numpy(wf) if isinstance(wf, np.ndarray) else wf
        for wf in waveform_list
    ]
    
    # Concatenate along the time dimension (dim=1)
    waveform_concat = torch.cat(waveform_tensors, dim=1)
    
    # Compute cumulative start positions
    waveform_start = torch.cumsum(
        torch.tensor([0] + [wf.shape[1] for wf in waveform_tensors[:-1]], dtype=torch.long),
        dim=0
    )
    
    return waveform_concat, waveform_start
