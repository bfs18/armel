"""
Generation utilities for ARMel model.

This module provides common utilities for autoregressive generation.
"""
import torch

from contextlib import nullcontext
from typing import List, Callable
from transformers import StaticCache, DynamicCache
from transformers.cache_utils import Cache
from utils.logger import get_logger
from torch.nn.attention import sdpa_kernel, SDPBackend


logger = get_logger()


def create_cache_for_model(model, device: torch.device, dtype: torch.dtype) -> Cache:
    """Create appropriate cache based on model's attention implementation."""
    attn_impl = getattr(model.llm.model.config, '_attn_implementation', 'sdpa')

    if attn_impl == 'flash_attention_2':
        logger.debug("Using DynamicCache for flash_attention_2")
        return DynamicCache()
    else:
        logger.debug(f"Using StaticCache for {attn_impl}")
        return StaticCache(
            config=model.llm.config,
            max_cache_len=model.config.max_tokens,
            max_batch_size=1,
            device=device,
            dtype=dtype
        )


def logits_to_probs(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
) -> torch.Tensor:
    temperature = max(temperature, 1e-5)
    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1),
        dim=-1
    )

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample(probs_sort):
    """Does multinomial sampling without a cuda synchronization."""
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(logits, **sampling_kwargs):
    logits_last = logits[0, -1]
    probs = logits_to_probs(logits=logits_last, **sampling_kwargs)
    idx_sample = multinomial_sample(probs)
    return idx_sample, probs


def build_prefix(rfmel, generated: List[torch.Tensor], batch_size: int, device, dtype):
    """Concise prefix builder for mel generation."""
    if not rfmel.use_prefix:
        return None
    if len(generated) < rfmel.num_prefix_patches:
        zero_prefix = rfmel.build_zero_prefix(batch_size, device, dtype)
        if len(generated) == 0:
            return zero_prefix
        else:
            return torch.cat([zero_prefix] + generated, dim=2)[..., -(rfmel.num_prefix_patches * rfmel.patch_size):]
    return torch.cat(generated[-rfmel.num_prefix_patches:], dim=2)


def get_prompt_embedding(model, prompt_ids, prompt_waveform, prompt_waveform_patch_start):
    """Get prompt embedding by merging text and waveform embeddings."""
    if prompt_waveform is not None and prompt_waveform.numel() > 0:
        _, waveform_emb = model.get_waveform_embedding(prompt_waveform)
        prompt_emb = model._merge_waveform_to_text(prompt_ids, waveform_emb, prompt_waveform_patch_start)
    else:
        prompt_emb = model.get_text_embedding(prompt_ids)
    return prompt_emb


class NoiseManager:
    """Manages pre-allocated consecutive noise and provides slices for each patch."""
    def __init__(self, z_consecutive: torch.Tensor = None):
        self.z_consecutive = z_consecutive
        self.patch_idx = 0
    
    def get_noise_slice(self, patch_size: int) -> torch.Tensor:
        """Get noise slice for current patch and auto-increment index."""
        if self.z_consecutive is None:
            return None
        start_frame = self.patch_idx * patch_size
        end_frame = (self.patch_idx + 1) * patch_size
        z_patch = self.z_consecutive[:, :, start_frame:end_frame]
        self.patch_idx += 1
        return z_patch


def decode_n_patches(
    model,
    first_emb: torch.Tensor,
    cache: StaticCache,
    decode_func: Callable,
    max_new_tokens: int,
    audio_eos_id: int,
    init_history: List[torch.Tensor] = None,
    noise_manager: NoiseManager = None,
    **sampling_kwargs
):
    """Decode multiple patches autoregressively."""
    history = list(init_history) if init_history is not None else []
    
    cur_emb = first_emb
    for i in range(max_new_tokens):
        with (sdpa_kernel(SDPBackend.MATH)
              if torch.cuda.is_available() else nullcontext()):
            cache_position = torch.tensor([cache.get_seq_length()], dtype=torch.long, device=cur_emb.device)
            out, new_cache = decode_func(
                model=model,
                emb=cur_emb,
                generated=history,
                cache=cache,
                cache_position=cache_position,
                noise_manager=noise_manager,
                **sampling_kwargs
            )
            cur_token = out['token']
            cur_comp_spec = out['comp_spec']
            cur_comp_spec, cur_emb = model.get_waveform_embedding_one_step(cur_comp_spec)
            cur_emb = cur_emb.transpose(1, 2)
            cache = new_cache
            if cur_token.item() == audio_eos_id:
                break
            history.append(cur_comp_spec)
    return history, cache
