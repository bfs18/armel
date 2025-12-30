#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import click
import torch
import torchaudio
import tqdm
import numpy as np
from typing import Optional
from lightning.pytorch import seed_everything

from ar.generate import get_prompt_embedding, build_prefix
from ar.mel_generate import generate, decode_one_patch
from ar.special_tokens import audio_eos_token, audio_out_token, pad_token
from ar.load_model import load_armel_for_inference
from utils.audio import read_audio, concat_waveforms_with_offsets
from utils.logger import get_logger
from dataset.data_types import Message, AudioContent
from dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from dataset.data_collate import SampleCollator
from utils.data import place_data
from einops import rearrange

PROJ_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = PROJ_DIR / "example_data" / "voice_prompts"

logger = get_logger()

# Global cache for vocos model to avoid reloading
_vocos_cache = {}


def get_or_load_vocos(
    device: str = "cuda",
    vocos_model_name: str = "charactr/vocos-mel-24khz"
):
    """
    Get or load vocos vocoder model with caching.
    
    Args:
        device: Device to run vocos on (cuda/cpu/mps)
        vocos_model_name: Name of the vocos model to use
    
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
        import traceback
        logger.error(traceback.format_exc())
        return None


def _mel_to_image(mel: torch.Tensor) -> np.ndarray:
    """Create a spectrogram image (RGB) for mel like rfwave_resample_lightning_module._mel_to_image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel_tensor = mel.detach().cpu().float()
    if mel_tensor.ndim == 3 and mel_tensor.shape[0] == 1:
        mel_tensor = mel_tensor.squeeze(0)
    mel_np = mel_tensor.numpy()

    height_inches = max(mel_np.shape[0] / 100.0, 1.0)
    width_inches = max(mel_np.shape[1] / 100.0, 1.0)

    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    ax.imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))
    plt.close(fig)
    return image


def reconstruct_waveform_from_mel(
    mel: torch.Tensor,
    output_path: str,
    sample_rate: int,
    device: str = "cuda",
    vocos_model_name: str = "charactr/vocos-mel-24khz",
    vocos: Optional[torch.nn.Module] = None
) -> Optional[torch.Tensor]:
    """
    Reconstruct waveform from mel spectrogram using vocos vocoder.
    
    Args:
        mel: Mel spectrogram tensor in log mel format, shape (B, n_mels, T) or (n_mels, T)
        output_path: Path to save the reconstructed audio file
        sample_rate: Sample rate of the audio
        device: Device to run vocos on (cuda/cpu/mps)
        vocos_model_name: Name of the vocos model to use (ignored if vocos is provided)
        vocos: Pre-loaded vocos model instance (optional, will load if None)
    
    Returns:
        Reconstructed waveform tensor if successful, None otherwise.
        Shape: (B, T) or (T,)
    """
    logger.info("=" * 80)
    logger.info("Reconstructing waveform with vocos...")
    logger.info("=" * 80)
    
    # Get or load vocos model
    if vocos is None:
        vocos = get_or_load_vocos(device=device, vocos_model_name=vocos_model_name)
        if vocos is None:
            return None
    
    try:
        # Ensure mel has batch dimension: (B, n_mels, T)
        mel_tensor = mel
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        
        # Mel is already in log mel format (from revert_norm_mel in generate function)
        # Shape: (B, n_mels, T)
        logger.info(f"Mel shape for vocos: {mel_tensor.shape}")
        
        # Move mel to device if needed
        mel_for_vocos = mel_tensor.to(device)
        
        # Reconstruct waveform
        logger.info("Decoding mel to waveform...")
        with torch.no_grad():
            reconstructed_audio = vocos.decode(mel_for_vocos.float())  # vocos is a nn.Module of float32.
        
        # Save audio
        audio_path = Path(output_path)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        reconstructed_audio_cpu = reconstructed_audio.cpu()
        
        torchaudio.save(
            str(audio_path),
            reconstructed_audio_cpu,
            sample_rate=sample_rate
        )
        logger.info(f"Saved reconstructed audio to: {audio_path}")
        
        # Stats
        audio_duration = reconstructed_audio_cpu.shape[1] / sample_rate
        logger.info(f"Reconstructed audio duration: {audio_duration:.3f} seconds")
        logger.info(f"Audio range: [{reconstructed_audio_cpu.min().item():.4f}, {reconstructed_audio_cpu.max().item():.4f}]")
        
        return reconstructed_audio_cpu
        
    except Exception as e:
        logger.error(f"Failed to reconstruct waveform with vocos: {e}")
        logger.error("Skipping waveform reconstruction.")
        import traceback
        logger.error(traceback.format_exc())
        return None


@torch.inference_mode()
def reconstruct_prompt_mel(model, prompt_ids, prompt_hidden_states, audio_out_id, prompt_waveform=None):
    """Reconstruct mel from prompt using hybrid approach:
    - Prefix: from generated patches (autoregressive, avoids discontinuities)
    - Skip features: from generated patches (autoregressive, avoids conflict with prefix)
    
    Args:
        model: ARMel model
        prompt_ids: token ids (1, T) or (T,)
        prompt_hidden_states: LLM hidden states (1, T, H)
        audio_out_id: audio_out token id
        prompt_waveform: ground truth waveform (optional, unused for skip features now)
    
    Returns:
        mel: reconstructed mel spectrogram
    """
    import torch.nn.functional as F
    assert prompt_hidden_states.dim() == 3 and prompt_hidden_states.size(0) == 1
    # prompt_ids can be (1, T) or (T,)
    if prompt_ids.dim() == 2:
        assert prompt_ids.size(0) == 1
        prompt_ids = prompt_ids.squeeze(0)
    audio_out_mask = (prompt_ids == audio_out_id)  # (T,)
    audio_out_mask_shift = F.pad(audio_out_mask, [0, 1])[..., 1:]  # (T,)

    # Select hidden states at positions that predict audio
    hs = prompt_hidden_states[:, audio_out_mask_shift, :]  # (1, N, H)
    if hs.size(1) == 0:
        return None
    
    # Generate patches one by one
    audio_hidden_states = hs.squeeze(0).unsqueeze(1)  # (N, 1, H)
    n_patches = audio_hidden_states.size(0)
    generated = []
    
    for i in range(n_patches):
        hidden = audio_hidden_states[i:i+1]  # (1, 1, H)
        
        # Build prefix from GENERATED patches (not ground truth) to avoid discontinuities
        prefix = build_prefix(model.rfmel, generated, batch_size=1, device=hidden.device, dtype=hidden.dtype)
        
        # Determine skip_features
        skip_features = None
        if model.use_skip_connection:
            if i == 0:
                # First patch: skip_features is 0
                skip_features = torch.zeros(1, model.resample_module.llm_hidden_dim, 1,
                                            device=hidden.device, dtype=hidden.dtype)
            else:
                # Compute from previous generated patch
                _, emb = model.get_waveform_embedding_one_step(generated[-1])
                skip_features = emb
        
        # Generate current patch
        comp_spec = model.forward_generate_wave(hidden, prefix=prefix, skip_features=skip_features)
        generated.append(comp_spec)
    
    comp_spec = torch.cat(generated, dim=2)  # (1, c, total_frames)
    mel = model.mel_processor.revert_norm_mel(comp_spec)
    return mel

def prepare_chunk_text(text: str, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1):
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER_"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        import langid
        import jieba
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for paragraph in paragraphs:
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            if chunks:
                chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def prepare_generation_context(ref_audio: str, model_sample_rate: int):
    messages = []
    waveform_l = []
    for ref_name in ref_audio.split(","):
        prompt_audio_path = PROMPT_DIR / f"{ref_name}.wav"
        prompt_text_path = PROMPT_DIR / f"{ref_name}.txt"
        assert prompt_audio_path.exists(), f"Voice prompt audio file {prompt_audio_path} does not exist."
        assert prompt_text_path.exists(), f"Voice prompt text file {prompt_text_path} does not exist."
        with open(prompt_text_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        waveform, _ = read_audio(prompt_audio_path, model_sample_rate)
        waveform = torch.from_numpy(waveform).float()
        waveform_l.append(waveform)

        messages.append(Message(role="user", content=prompt_text))
        messages.append(Message(role="assistant", content=AudioContent(audio_url=str(prompt_audio_path))))
    return messages, waveform_l


@torch.inference_mode()
def run_mel_inference(
    model,
    tokenizer,
    collator,
    prompt_messages,
    prompt_waveforms,
    chunked_text,
    max_new_tokens,
    top_p: float,
    temperature: float,
):
    audio_eos_id = tokenizer.convert_tokens_to_ids(audio_eos_token)
    audio_out_id = tokenizer.convert_tokens_to_ids(audio_out_token)
    generated_mels = []
    generation_messages = []
    reconstructed_prompt_mel = None

    for idx, chunk_text in tqdm.tqdm(enumerate(chunked_text), desc="Generating mel chunks", total=len(chunked_text)):
        generation_messages.append(Message(role="user", content=chunk_text))
        chatml_sample = ChatMLSample(messages=prompt_messages + generation_messages)
        input_tokens, *_ = prepare_chatml_sample(chatml_sample, tokenizer)
        postfix = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>", add_special_tokens=False)
        input_tokens.extend(postfix)

        logger.info(f"========= Chunk {idx} Input =========")
        logger.info(tokenizer.decode(input_tokens))

        # Only use prompts as conditioning for mel pipeline
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
        prompt_emb = get_prompt_embedding(model, prompt_data.input_ids, prompt_data.waveform, prompt_data.waveform_patch_start)

        mel, prompt_hidden_states = generate(
            model=model,
            prompt_emb=prompt_emb,
            decode_func=decode_one_patch,
            max_new_tokens=max_new_tokens,
            audio_eos_id=audio_eos_id,
            top_p=top_p,
            temperature=temperature,
        )
        generated_mels.append(mel.to('cpu'))
        if idx == 0:
            prompt_ids = prompt_data.input_ids.to(prompt_hidden_states.device)
            reconstructed_prompt_mel = reconstruct_prompt_mel(
                model, prompt_ids, prompt_hidden_states, audio_out_id, 
                prompt_waveform=prompt_data.waveform
            )

    if len(generated_mels) == 0:
        raise RuntimeError("No mel generated.")
    mel_out = torch.cat(generated_mels, dim=2)
    return mel_out, reconstructed_prompt_mel


@click.command()
@click.option('--model_path', type=str, required=True, help='Path to exported model (.ckpt) or directory with model files')
@click.option('--text', type=str, required=True, help='Text to synthesize')
@click.option('--ref_audio', type=str, default='fanren09', help='Reference audio name(s) from example_data/voice_prompts, comma-separated')
@click.option('--output_path', type=str, default='mel_output', help='Output path prefix (without extension). Files will be saved as prefix.png, prefix.npy, prefix.wav, etc.')
@click.option('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'bfloat16']), default='bfloat16', help='Data type')
@click.option('--max_new_tokens', type=int, default=1024, help='Max new tokens to generate')
@click.option('--top_p', type=float, default=0.7, help='Top-p sampling')
@click.option('--temperature', type=float, default=0.7, help='Temperature')
@click.option('--chunk_method', type=click.Choice(['speaker', 'word', 'none']), default='speaker', help='Text chunk method')
@click.option('--chunk_max_word_num', type=int, default=100, help='Max words per chunk (word method)')
@click.option('--chunk_max_num_turns', type=int, default=1, help='Max turns per chunk (speaker method)')
@click.option('--seed', type=int, default=42, help='Random seed')
def main(
    model_path: str,
    text: str,
    ref_audio: str,
    output_path: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    chunk_method: str,
    chunk_max_word_num: int,
    chunk_max_num_turns: int,
    seed: int,
):
    seed_everything(seed)
    
    logger.info("=" * 80)
    logger.info("ARMel Inference (mel spectrogram output)")
    logger.info(f"Random seed: {seed}")
    logger.info("=" * 80)

    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_armel_for_inference(model_path=model_path, device=device, dtype=torch_dtype)

    # Collator using MelConfig from model
    mel_config = model.mel_processor.config
    audio_out_token_id = tokenizer.convert_tokens_to_ids(audio_out_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    collator = SampleCollator(
        audio_out_token_id=audio_out_token_id,
        pad_token_id=pad_token_id,
        head_config=mel_config,
        patch_size=model.config.patch_size,
        round_to=1,
    )

    # Load ref audio and text chunks
    logger.info(f"Loading reference audio: {ref_audio}")
    prompt_messages, prompt_waveforms = prepare_generation_context(ref_audio=ref_audio, model_sample_rate=model.config.sample_rate)

    if os.path.exists(text):
        logger.info(f"Loading transcript from {text}")
        with open(text, "r", encoding="utf-8") as f:
            text = f.read().strip()
    chunk_method_value = None if chunk_method == 'none' else chunk_method
    chunked_text = prepare_chunk_text(text=text, chunk_method=chunk_method_value, chunk_max_word_num=chunk_max_word_num, chunk_max_num_turns=chunk_max_num_turns)
    logger.info(f"Text split into {len(chunked_text)} chunks")

    # Generate mel
    logger.info("Generating mel spectrogram...")
    mel, recon_prompt_mel = run_mel_inference(
        model=model,
        tokenizer=tokenizer,
        collator=collator,
        prompt_messages=prompt_messages,
        prompt_waveforms=prompt_waveforms,
        chunked_text=chunked_text,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
    )

    # Use unified output path prefix
    base_path = Path(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-load vocos once (for both generated and reconstructed prompt)
    vocos = get_or_load_vocos(device=device)
    
    # Save generated mel outputs
    # 1. Image: output_path.png
    img_path = base_path.with_suffix('.png')
    img = _mel_to_image(mel.squeeze(0))  # (1, n_mels, T) -> (n_mels, T)
    try:
        from PIL import Image
        Image.fromarray(img).save(img_path)
    except Exception:
        import imageio
        imageio.imwrite(str(img_path), img)
    logger.info(f"Saved mel spectrogram image to: {img_path}")
    
    # 2. NPY: output_path.npy
    npy_path = base_path.with_suffix('.npy')
    np.save(npy_path, mel.squeeze(0).detach().cpu().float().numpy())
    logger.info(f"Saved mel array to: {npy_path}")
    
    # 3. Audio: output_path.wav
    audio_path = base_path.with_suffix('.wav')
    reconstruct_waveform_from_mel(
        mel=mel,
        output_path=str(audio_path),
        sample_rate=model.config.sample_rate,
        device=device,
        vocos=vocos
    )
    
    # Save reconstructed prompt outputs with _recon_prompt suffix
    if recon_prompt_mel is not None:
        # 1. Image: output_path_recon_prompt.png
        recon_img_path = base_path.with_name(base_path.stem + '_recon_prompt').with_suffix('.png')
        recon_img = _mel_to_image(recon_prompt_mel.squeeze(0))
        try:
            from PIL import Image
            Image.fromarray(recon_img).save(recon_img_path)
        except Exception:
            import imageio
            imageio.imwrite(str(recon_img_path), recon_img)
        logger.info(f"Saved reconstructed prompt mel image to: {recon_img_path}")
        
        # 2. NPY: output_path_recon_prompt.npy
        recon_npy_path = base_path.with_name(base_path.stem + '_recon_prompt').with_suffix('.npy')
        np.save(recon_npy_path, recon_prompt_mel.squeeze(0).detach().cpu().float().numpy())
        logger.info(f"Saved reconstructed prompt mel array to: {recon_npy_path}")
        
        # 3. Audio: output_path_recon_prompt.wav (using vocos)
        recon_audio_path = base_path.with_name(base_path.stem + '_recon_prompt').with_suffix('.wav')
        reconstruct_waveform_from_mel(
            mel=recon_prompt_mel,
            output_path=str(recon_audio_path),
            sample_rate=model.config.sample_rate,
            device=device,
            vocos=vocos
        )
    
    # Quick stats
    n_mels = mel.size(1)
    n_frames = mel.size(2)
    hop = mel_config.hop_length or 1
    duration_s = (n_frames * hop) / model.config.sample_rate
    logger.info("=" * 80)
    logger.info(f"Mel shape: (n_mels={n_mels}, frames={n_frames})  ~ {duration_s:.2f}s")
    logger.info("Inference complete.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
