import torch

from ar.armel import ARMel
from typing import Callable
from utils.logger import get_logger
from functools import partial
from ar.generate import decode_n_patches, create_cache_for_model


logger = get_logger()


def build_prefix_mel(rfmel, generated, batch_size: int, device, dtype):
    if not rfmel.use_prefix:
        return None
    if len(generated) < rfmel.num_prefix_patches:
        c = rfmel.estimator.output_channels
        l = (rfmel.num_prefix_patches - len(generated)) * rfmel.patch_size
        zero_prefix = torch.zeros(batch_size, c, l, device=device, dtype=dtype)
        return zero_prefix if len(generated) == 0 else torch.cat([zero_prefix] + generated, dim=2)
    return torch.cat(generated[-rfmel.num_prefix_patches:], dim=2)


def decode_one_patch(
    model: ARMel,
    emb: torch.Tensor,
    generated,
    cache=None,
    cache_position=None,
    is_first: bool = False,
    noise_manager=None,  # Not used for mel, but required by decode_n_patches signature
    **sampling_kwargs
):
    """Decode one patch of mel spectrogram.
    
    Args:
        model: ARMel model
        emb: input embedding (B, 1, hidden_dim) - contains downsample output from previous patch
        generated: list of previously generated patches
        cache: KV cache for LLM
        cache_position: position in cache
        is_first: whether this is the first patch
        noise_manager: not used for mel
        **sampling_kwargs: additional sampling arguments
    
    Returns:
        res: dict with 'token' and 'comp_spec'
        new_cache: updated KV cache
    """
    out, new_cache = model.forward_generate_ar(emb, cache, cache_position=cache_position, return_all=is_first)
    if is_first:
        token_logits = out['token_logits'][:, -1:]
        hidden_states = out['hidden_states'][:, -1:]
    else:
        token_logits = out['token_logits']
        hidden_states = out['hidden_states']
    # Sample next token
    from ar.generate import sample  # reuse shared sampler
    token, _ = sample(token_logits, **sampling_kwargs)

    b = hidden_states.size(0)
    prefix = build_prefix_mel(model.rfmel, generated, batch_size=b, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Use emb as skip_features if skip connection is enabled
    # emb already contains downsample output from previous patch
    skip_features = None
    if model.use_skip_connection:
        if is_first:
            # BOS: emb is from text, not audio, use zeros
            skip_features = torch.zeros(b, model.resample_module.llm_hidden_dim, 1, 
                                       device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            # emb: (B, 1, hidden_dim) -> skip_features: (B, hidden_dim, 1)
            skip_features = emb.transpose(1, 2)
    
    comp_spec = model.forward_generate_wave(hidden_states, prefix=prefix, skip_features=skip_features)
    res = {"token": token, "comp_spec": comp_spec}
    if is_first:
        res['all_hidden_states'] = out['hidden_states']
    return res, new_cache



@torch.inference_mode()
def generate(
        model: ARMel,
        prompt_emb: torch.Tensor,
        decode_func: Callable,
        max_new_tokens: int,
        audio_eos_id: int,
        **sampling_kwargs
):
    T = prompt_emb.size(1)
    if max_new_tokens:
        if T + max_new_tokens > model.config.max_tokens:
            max_new_tokens = model.config.max_tokens - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")
    else:
        T_new = model.config.max_tokens
        max_new_tokens = T_new - T

    device, dtype = prompt_emb.device, prompt_emb.dtype

    cache = create_cache_for_model(model, device, next(model.parameters()).dtype)

    prefill_func = partial(decode_one_patch, is_first=True)
    out, cache = prefill_func(
        model,
        prompt_emb,
        generated=[],
        cache=cache,
        **sampling_kwargs
    )

    prompt_hidden_states = out['all_hidden_states']
    # First mel patch generated from audio_out_bos position
    first_comp_spec = out['comp_spec']

    # remove left and right padding frames in get_waveform_embedding_one_step.
    first_comp_spec, first_emb = model.get_waveform_embedding_one_step(first_comp_spec)
    # Ensure single-token step embedding: (B, 1, hidden_dim)
    first_emb = first_emb.transpose(1, 2)
    history, cache = decode_n_patches(
        model=model,
        first_emb=first_emb,
        cache=cache,
        decode_func=decode_func,
        max_new_tokens=max_new_tokens,
        audio_eos_id=audio_eos_id,
        init_history=[first_comp_spec],
        **sampling_kwargs
    )
    # Concatenate all patches in history (already includes first_comp_spec)
    comp_spec = torch.cat(history, dim=2)
    mel = model.mel_processor.revert_norm_mel(comp_spec)
    return mel, prompt_hidden_states
