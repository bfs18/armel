"""
Shared inference test utilities for Lightning modules.
Provides functions to run inference and log audio to wandb during validation.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List

from ar.special_tokens import audio_eos_token, audio_out_token, pad_token
from utils.audio import read_audio, concat_waveforms_with_offsets
from utils.logger import get_logger
from dataset.data_types import Message, AudioContent
from dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from dataset.data_collate import SampleCollator
from utils.data import place_data

logger = get_logger(__name__)

# Global cache for vocos model to avoid reloading
_vocos_cache = {}


def get_or_load_vocos(device: str = "cuda", vocos_model_name: str = "charactr/vocos-mel-24khz"):
    """Get cached vocos model or load it if not cached.

    Args:
        device: Device to load the model on
        vocos_model_name: Name of the vocos model to load

    Returns:
        Vocos model instance if successful, None otherwise
    """
    cache_key = f"{vocos_model_name}_{device}"

    if cache_key in _vocos_cache:
        return _vocos_cache[cache_key]

    try:
        from vocos import Vocos
        logger.info(f"Loading vocos vocoder: {vocos_model_name}")
        vocos = Vocos.from_pretrained(vocos_model_name)
        vocos = vocos.to(device)
        vocos.eval()
        _vocos_cache[cache_key] = vocos
        logger.info(f"Vocos vocoder loaded and cached for {device}")
        return vocos
    except ImportError:
        logger.error("vocos package not found. Please install it with: pip install vocos")
        return None
    except Exception as e:
        logger.error(f"Failed to load vocos vocoder: {e}")
        return None

# Default paths
PROJ_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJ_DIR / "example_data" / "voice_prompts"
TRANSCRIPT_DIR = PROJ_DIR / "example_data" / "transcript"

# Default inference test config
DEFAULT_REF_AUDIO = "fanren08"
DEFAULT_TRANSCRIPT = "fanren_short.txt"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TOP_P = 0.7
DEFAULT_TEMPERATURE = 0.7


def prepare_generation_context(ref_audio: str, model_sample_rate: int) -> Tuple[List[Message], List[torch.Tensor]]:
    """Prepare prompt messages and waveforms from reference audio."""
    messages = []
    waveform_l = []

    for ref_name in ref_audio.split(","):
        prompt_audio_path = PROMPT_DIR / f"{ref_name}.wav"
        prompt_text_path = PROMPT_DIR / f"{ref_name}.txt"
        if not prompt_audio_path.exists() or not prompt_text_path.exists():
            logger.warning(f"Reference audio {ref_name} not found, skipping inference test")
            return [], []
        with open(prompt_text_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        waveform, _ = read_audio(prompt_audio_path, model_sample_rate)
        waveform = torch.from_numpy(waveform).float()
        waveform_l.append(waveform)

        messages.append(Message(role="user", content=prompt_text))
        messages.append(Message(role="assistant", content=AudioContent(audio_url=str(prompt_audio_path))))

    return messages, waveform_l


def load_transcript(transcript_name: str) -> Optional[str]:
    """Load transcript text from file."""
    transcript_path = TRANSCRIPT_DIR / transcript_name
    if not transcript_path.exists():
        logger.warning(f"Transcript {transcript_name} not found")
        return None
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read().strip()


@torch.inference_mode()
def reconstruct_prompt_waveform(model, prompt_ids, prompt_hidden_states, audio_out_id):
    """Reconstruct waveform from prompt hidden states.

    Args:
        model: ARWave model
        prompt_ids: token ids (1, T)
        prompt_hidden_states: LLM hidden states (1, T, H)
        audio_out_id: audio_out token id

    Returns:
        waveform: reconstructed waveform or None if no audio tokens
    """
    from ar.generate import build_prefix, NoiseManager

    assert prompt_hidden_states.size(0) == 1

    audio_out_mask = prompt_ids == audio_out_id
    # left shift the mask to extract audio out hidden states
    audio_out_mask_shift = F.pad(audio_out_mask, [0, 1])[..., 1:]
    audio_hidden_states = prompt_hidden_states[audio_out_mask_shift]  # (N, H)
    if audio_hidden_states.size(0) == 0:
        return None

    # Generate patches one by one: (N, H) -> (N, 1, H)
    audio_hidden_states = audio_hidden_states.unsqueeze(1)
    n_patches = audio_hidden_states.size(0)
    generated = []

    # Pre-allocate consecutive noise for all patches
    total_frames = n_patches * model.config.patch_size
    z_consecutive = model.rfwave.get_noise(
        batch_size=1,
        num_frames=total_frames,
        device=audio_hidden_states.device
    ).to(audio_hidden_states.dtype)
    noise_manager = NoiseManager(z_consecutive)

    for i in range(n_patches):
        hidden = audio_hidden_states[i:i+1]  # (1, 1, H)
        prefix = build_prefix(model.rfwave, generated, batch_size=1, device=hidden.device, dtype=hidden.dtype)

        skip_features = None
        if model.use_skip_connection:
            if i == 0:
                skip_features = torch.zeros(1, model.resample_module.llm_hidden_dim, 1,
                                            device=hidden.device, dtype=hidden.dtype)
            else:
                _, emb = model.get_waveform_embedding_one_step(generated[-1])
                skip_features = emb

        z = noise_manager.get_noise_slice(model.config.patch_size)
        comp_spec = model.forward_generate_wave(hidden, prefix=prefix, z=z, skip_features=skip_features)
        generated.append(comp_spec)

    comp_spec = torch.cat(generated, dim=2)
    waveform = model.wave_processor.get_wave(comp_spec)
    return waveform


@torch.inference_mode()
def run_arwave_inference_test(
    model,
    tokenizer,
    ref_audio: str = DEFAULT_REF_AUDIO,
    transcript: str = DEFAULT_TRANSCRIPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run inference test for ARWave model and return generated waveform.

    Returns:
        Tuple of (generated_waveform, reconstructed_prompt_waveform)
    """
    from ar.generate import get_prompt_embedding, decode_one_patch, generate

    # Prepare context
    prompt_messages, prompt_waveforms = prepare_generation_context(ref_audio, model.config.sample_rate)
    if not prompt_messages:
        return None, None

    text = load_transcript(transcript)
    if text is None:
        return None, None

    # Create collator
    head_config = model.wave_processor.head_config
    audio_out_token_id = tokenizer.convert_tokens_to_ids(audio_out_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    audio_eos_id = tokenizer.convert_tokens_to_ids(audio_eos_token)

    collator = SampleCollator(
        audio_out_token_id=audio_out_token_id,
        pad_token_id=pad_token_id,
        head_config=head_config,
        patch_size=model.config.patch_size,
        round_to=1
    )

    # Prepare input
    generation_messages = [Message(role="user", content=text)]
    chatml_sample = ChatMLSample(messages=prompt_messages + generation_messages)
    input_tokens, *_ = prepare_chatml_sample(chatml_sample, tokenizer)
    postfix = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>", add_special_tokens=False)
    input_tokens.extend(postfix)

    waveform_concat, waveform_start = concat_waveforms_with_offsets(prompt_waveforms)
    prompt_sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        label_ids=None,
        audio_waveforms_concat=waveform_concat,
        audio_waveforms_start=waveform_start,
        audio_sample_rate=torch.tensor([model.config.sample_rate] * len(waveform_start)),
        audio_speaker_indices=None,
    )
    prompt_data = collator([prompt_sample])
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    prompt_data = place_data(prompt_data, device=model_device, dtype=model_dtype)

    # Generate
    prompt_emb = get_prompt_embedding(model, prompt_data.input_ids, prompt_data.waveform, prompt_data.waveform_patch_start)
    waveform, prompt_hidden_states = generate(
        model=model,
        prompt_emb=prompt_emb,
        decode_func=decode_one_patch,
        max_new_tokens=max_new_tokens,
        audio_eos_id=audio_eos_id,
        top_p=top_p,
        temperature=temperature
    )

    # Reconstruct prompt waveform
    reconstructed_prompt = reconstruct_prompt_waveform(
        model, prompt_data.input_ids, prompt_hidden_states, audio_out_token_id
    )

    return waveform, reconstructed_prompt


@torch.inference_mode()
def reconstruct_prompt_mel(model, prompt_ids, prompt_hidden_states, audio_out_id):
    """Reconstruct mel from prompt hidden states.

    Args:
        model: ARMel model
        prompt_ids: token ids (1, T)
        prompt_hidden_states: LLM hidden states (1, T, H)
        audio_out_id: audio_out token id

    Returns:
        mel: reconstructed mel spectrogram or None if no audio tokens
    """
    from ar.mel_generate import build_prefix_mel

    assert prompt_hidden_states.size(0) == 1

    audio_out_mask = prompt_ids == audio_out_id
    # left shift the mask to extract audio out hidden states
    audio_out_mask_shift = F.pad(audio_out_mask, [0, 1])[..., 1:]
    audio_hidden_states = prompt_hidden_states[audio_out_mask_shift]  # (N, H)
    if audio_hidden_states.size(0) == 0:
        return None

    # Generate patches one by one: (N, H) -> (N, 1, H)
    audio_hidden_states = audio_hidden_states.unsqueeze(1)
    n_patches = audio_hidden_states.size(0)
    generated = []

    for i in range(n_patches):
        hidden = audio_hidden_states[i:i+1]  # (1, 1, H)
        prefix = build_prefix_mel(model.rfmel, generated, batch_size=1, device=hidden.device, dtype=hidden.dtype)

        skip_features = None
        if model.use_skip_connection:
            if i == 0:
                skip_features = torch.zeros(1, model.resample_module.llm_hidden_dim, 1,
                                            device=hidden.device, dtype=hidden.dtype)
            else:
                _, emb = model.get_waveform_embedding_one_step(generated[-1])
                skip_features = emb

        comp_spec = model.forward_generate_wave(hidden, prefix=prefix, skip_features=skip_features)
        generated.append(comp_spec)

    comp_spec = torch.cat(generated, dim=2)
    mel = model.mel_processor.revert_norm_mel(comp_spec)
    return mel


@torch.inference_mode()
def run_armel_inference_test(
    model,
    tokenizer,
    ref_audio: str = DEFAULT_REF_AUDIO,
    transcript: str = DEFAULT_TRANSCRIPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run inference test for ARMel model and return generated waveform.

    Returns:
        Tuple of (generated_waveform, reconstructed_prompt_waveform)
    """
    from ar.generate import get_prompt_embedding
    from ar.mel_generate import generate, decode_one_patch

    # Prepare context
    prompt_messages, prompt_waveforms = prepare_generation_context(ref_audio, model.config.sample_rate)
    if not prompt_messages:
        return None, None

    text = load_transcript(transcript)
    if text is None:
        return None, None

    # Create collator
    mel_config = model.mel_processor.config
    audio_out_token_id = tokenizer.convert_tokens_to_ids(audio_out_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    audio_eos_id = tokenizer.convert_tokens_to_ids(audio_eos_token)

    collator = SampleCollator(
        audio_out_token_id=audio_out_token_id,
        pad_token_id=pad_token_id,
        head_config=mel_config,
        patch_size=model.config.patch_size,
        round_to=1
    )

    # Prepare input
    generation_messages = [Message(role="user", content=text)]
    chatml_sample = ChatMLSample(messages=prompt_messages + generation_messages)
    input_tokens, *_ = prepare_chatml_sample(chatml_sample, tokenizer)
    postfix = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>", add_special_tokens=False)
    input_tokens.extend(postfix)

    waveform_concat, waveform_start = concat_waveforms_with_offsets(prompt_waveforms)
    prompt_sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        label_ids=None,
        audio_waveforms_concat=waveform_concat,
        audio_waveforms_start=waveform_start,
        audio_sample_rate=torch.tensor([model.config.sample_rate] * len(waveform_start)),
        audio_speaker_indices=None,
    )
    prompt_data = collator([prompt_sample])
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    prompt_data = place_data(prompt_data, device=model_device, dtype=model_dtype)

    # Generate mel
    prompt_emb = get_prompt_embedding(model, prompt_data.input_ids, prompt_data.waveform, prompt_data.waveform_patch_start)
    mel, prompt_hidden_states = generate(
        model=model,
        prompt_emb=prompt_emb,
        decode_func=decode_one_patch,
        max_new_tokens=max_new_tokens,
        audio_eos_id=audio_eos_id,
        top_p=top_p,
        temperature=temperature
    )

    # Convert mel to waveform using cached vocos
    vocos = get_or_load_vocos(device=str(model_device))
    if vocos is None:
        logger.warning("Vocos not available, cannot convert mel to waveform")
        return None, None

    with torch.no_grad():
        mel = mel.float()  # Ensure float32 for vocos
        waveform = vocos.decode(mel.to(model_device))

    # Reconstruct prompt mel and convert to waveform
    reconstructed_prompt = None
    reconstructed_mel = reconstruct_prompt_mel(
        model, prompt_data.input_ids, prompt_hidden_states, audio_out_token_id
    )
    if reconstructed_mel is not None:
        with torch.no_grad():
            reconstructed_mel = reconstructed_mel.float()
            reconstructed_prompt = vocos.decode(reconstructed_mel.to(model_device))

    return waveform.cpu(), reconstructed_prompt.cpu() if reconstructed_prompt is not None else None

