"""
Load ARMel model for inference.

Usage:
    from ar.load_model import load_armel_for_inference
    
    model, tokenizer = load_armel_for_inference(
        model_path="model.ckpt",  # Will auto-load model.yaml
        device="cuda"
    )
"""
import torch
from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional, Tuple

from ar.armel import ARMel
from ar.armel_config import ARMelConfig
from ar.qwen import Qwen3LM
from ar.qwen_tokenizer import get_qwen_tokenizer
from rfwave.mel_processor import MelProcessor, MelConfig
from rfwave.resample import ResampleModule
from rfwave.estimator import RFBackbone
from rfwave.mel_model import RFMelConfig
from utils.config_utils import get_rfmel_config_defaults
from utils.logger import get_logger

logger = get_logger(__name__)


def load_armel_for_inference(
    model_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None
) -> Tuple[ARMel, any]:
    """
    Load ARMel (mel-spectrogram) model for inference from an exported checkpoint.
    Expects a matching model.yaml alongside the checkpoint.
    
    Args:
        model_path: Path to model weights file (.ckpt) or directory
        device: Device ('cuda', 'cpu', etc.)
        dtype: Data type (None to use default)
    
    Returns:
        model: ARMel model
        tokenizer: Tokenizer
    """
    model_path = Path(model_path)

    # Resolve weights and config paths
    if model_path.is_dir():
        weights_path = model_path / "model.ckpt"
        config_path = model_path / "model.yaml"
    else:
        weights_path = model_path
        config_path = weights_path.with_suffix('.yaml')

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("=" * 80)
    logger.info("Loading ARMel model for inference")
    logger.info("=" * 80)

    # Load config
    logger.info(f"Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Force SDPA for inference to enable StaticCache
    # Flash Attention 2 doesn't support StaticCache, but SDPA does
    # StaticCache is more efficient for inference than DynamicCache
    attn_impl = cfg.model.get('attn_implementation', 'sdpa')
    if attn_impl == 'flash_attention_2':
        logger.info(f"Changing attention from '{attn_impl}' to 'sdpa' for inference (StaticCache compatibility)")
        cfg.model.attn_implementation = 'sdpa'
    else:
        logger.info(f"Using attention implementation: {attn_impl}")

    # Load checkpoint
    logger.info(f"Loading weights: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Invalid weights file format, please use mel_export_checkpoint.py")
    model_state_dict = checkpoint['model_state_dict']

    # Tokenizer and LLM
    # For inference, we only need the tokenizer and model config from llm_model_path
    # The actual LLM weights will be loaded from the checkpoint
    tokenizer = get_qwen_tokenizer(cfg.model.llm_model_path)
    llm = Qwen3LM(
        pretrain_path=cfg.model.llm_model_path,
        load_weights=False,  # Don't load pretrained weights, will load from checkpoint
        attn_implementation=cfg.model.attn_implementation
    )
    llm.resize_token_embeddings(len(tokenizer))

    # Mel processor
    mel_cfg = cfg.model.mel
    mel_config = MelConfig(
        sample_rate=mel_cfg.sample_rate,
        n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels,
        padding=mel_cfg.get('padding', 'same'),
    )
    mel_processor = MelProcessor(mel_config)

    comp_spec_dim = mel_processor.config.n_mels
    llm_hidden_dim = llm.model.config.hidden_size

    # Estimator
    est_cfg = cfg.model.estimator
    rfmel_cfg = cfg.model.get('rfmel', {})
    rfmel_defaults = get_rfmel_config_defaults()
    num_prefix_patches = int(rfmel_cfg.get('num_prefix_patches', rfmel_defaults['num_prefix_patches']))
    estimator = RFBackbone(
        input_channels=comp_spec_dim,
        output_channels=comp_spec_dim,
        dim=est_cfg.hidden_dim,
        intermediate_dim=est_cfg.intermediate_dim,
        num_layers=est_cfg.num_layers,
        prev_cond=bool(num_prefix_patches > 0),
    )

    # Resample module
    rs_cfg = cfg.model.resample
    resample_module = ResampleModule(
        complex_spec_dim=comp_spec_dim,
        llm_hidden_dim=llm_hidden_dim,
        patch_size=cfg.model.patch_size,
        resample_type=rs_cfg.get('resample_type', 'conv'),
        hidden_dims=rs_cfg.get('hidden_dims', 512),
        downsample_strides=rs_cfg.get('downsample_strides', [2, 4]),
        perceiver_depth=rs_cfg.get('perceiver_depth', 6),
        perceiver_heads=rs_cfg.get('perceiver_heads', 8),
        dim_scale_per_stage=rs_cfg.get('dim_scale_per_stage', 1.0),
    )

    # RFMel configuration
    rfmel_config = RFMelConfig(
        solver=rfmel_cfg.get('solver', rfmel_defaults['solver']),
        noise_std=rfmel_cfg.get('noise_std', rfmel_defaults['noise_std']),
        t_scheduler=rfmel_cfg.get('t_scheduler', rfmel_defaults['t_scheduler']),
        training_cfg_rate=rfmel_cfg.get('training_cfg_rate', rfmel_defaults['training_cfg_rate']),
        inference_cfg_rate=rfmel_cfg.get('inference_cfg_rate', rfmel_defaults['inference_cfg_rate']),
        n_timesteps=rfmel_cfg.get('n_timesteps', rfmel_defaults['n_timesteps']),
        batch_mul=rfmel_cfg.get('batch_mul', rfmel_defaults['batch_mul']),
        patch_size=cfg.model.patch_size,
        num_prefix_patches=num_prefix_patches,
    )

    # ARMel config
    armel_cfg = cfg.armel
    armel_config = ARMelConfig(
        audio_out_token=armel_cfg.audio_out_token,
        audio_out_bos_token=armel_cfg.audio_out_bos_token,
        audio_eos_token=armel_cfg.audio_eos_token,
        audio_out_token_id=tokenizer.convert_tokens_to_ids(armel_cfg.audio_out_token),
        audio_out_bos_token_id=tokenizer.convert_tokens_to_ids(armel_cfg.audio_out_bos_token),
        audio_eos_token_id=tokenizer.convert_tokens_to_ids(armel_cfg.audio_eos_token),
        ignore_index=armel_cfg.ignore_index,
        round_to=armel_cfg.round_to,
        max_tokens=armel_cfg.max_tokens,
        sample_rate=armel_cfg.sample_rate,
        patch_size=cfg.model.patch_size,
        use_skip_connection=bool(cfg.model.get('use_skip_connection', False))
    )

    # Create model
    model = ARMel(
        config=armel_config,
        llm=llm,
        mel_processor=mel_processor,
        estimator=estimator,
        resample_module=resample_module,
        rfmel_config=rfmel_config,
    )
    model.mel_processor.to(torch.float32)

    # Load weights
    model.load_state_dict(model_state_dict)
    
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("=" * 80)
    logger.info("ARMel model loaded successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    return model, tokenizer
