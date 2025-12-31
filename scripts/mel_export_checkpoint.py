#!/usr/bin/env python3
"""
Export ARMel (mel-spectrogram) model weights and config from a Lightning checkpoint.

Outputs a self-contained directory with:
  - model.ckpt  (weights)
  - model.yaml  (inference config with embedded LLM config)
  - tokenizer/  (tokenizer files)

Usage:
  python scripts/mel_export_checkpoint.py --ckpt_path logs_mel/checkpoints/last.ckpt --output_path exported_mel/
  python scripts/mel_export_checkpoint.py --ckpt_path logs_mel/checkpoints --output_path exported_mel/
"""
import argparse
import sys
import shutil
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from dataclasses import fields as dataclass_fields
from omegaconf import OmegaConf
from transformers import Qwen3ForCausalLM

from utils.logger import get_logger
from utils.checkpoints import find_latest_checkpoint
from rfwave.mel_model import RFMelConfig


logger = get_logger(__name__)


def _get_rfmel_config_defaults():
    """Return defaults from RFMelConfig dataclass with Enums converted to string values."""
    defaults = {}
    for f in dataclass_fields(RFMelConfig):
        default_value = f.default
        if hasattr(default_value, 'value'):
            default_value = default_value.value
        defaults[f.name] = default_value
    return defaults


def export_mel_checkpoint(ckpt_path: str, output_path: str):
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)

    # If a directory is provided, pick latest checkpoint
    if ckpt_path.is_dir():
        resolved_ckpt = find_latest_checkpoint(ckpt_path)
        if resolved_ckpt is None:
            raise FileNotFoundError(f"No checkpoint files found in directory: {ckpt_path}")
        ckpt_path = Path(resolved_ckpt)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    logger.info("=" * 80)
    logger.info(f"Exporting ARMel model from checkpoint: {ckpt_path}")
    logger.info("=" * 80)

    # Output must be a directory for self-contained export
    if output_path.suffix.lower() == ".ckpt":
        logger.warning("Output path should be a directory for self-contained export. Using parent directory.")
        output_dir = output_path.parent
        model_path = output_path
        config_path = output_path.with_suffix(".yaml")
    else:
        output_dir = output_path
        model_path = output_dir / "model.ckpt"
        config_path = output_dir / "model.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"

    # Load checkpoint
    logger.info("Loading checkpoint...")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

    global_step = checkpoint.get('global_step', 0)
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Checkpoint info: epoch={epoch}, global_step={global_step}")

    # Extract config for inference
    hyper_parameters = checkpoint.get('hyper_parameters', {})
    if 'cfg' not in hyper_parameters:
        raise ValueError("Config not found in checkpoint under hyper_parameters['cfg']")
    cfg = hyper_parameters['cfg']

    # Build armel (token/audio) config with dataset merged values
    armel_config = OmegaConf.to_container(cfg.arwave, resolve=True)
    armel_config['sample_rate'] = cfg.dataset.sample_rate
    armel_config['max_tokens'] = cfg.dataset.max_tokens

    # Model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)

    # Ensure rfmel has all defaults
    if 'rfmel' not in model_config:
        model_config['rfmel'] = {}
    rfmel_defaults = _get_rfmel_config_defaults()
    # Fill missing rfmel fields from dataclass defaults
    for key, default_value in rfmel_defaults.items():
        if key not in model_config['rfmel']:
            model_config['rfmel'][key] = default_value
            logger.info(f"Adding missing rfmel config: {key} = {default_value}")
    # Ensure patch_size in rfmel matches top-level model.patch_size
    model_config['rfmel']['patch_size'] = model_config.get('patch_size', model_config['rfmel']['patch_size'])

    # Export LLM config (instead of llm_model_path)
    logger.info(f"Loading LLM config from: {cfg.model.llm_model_path}")
    llm_config = Qwen3ForCausalLM.config_class.from_pretrained(cfg.model.llm_model_path)
    llm_config_dict = llm_config.to_dict()
    # Override attention implementation from training config
    llm_config_dict['attn_implementation'] = model_config.get('attn_implementation', 'sdpa')
    # Remove problematic fields that are None (causes issues during deserialization)
    problematic_fields = ['label2id', 'id2label']
    for field in problematic_fields:
        if field in llm_config_dict and llm_config_dict[field] is None:
            del llm_config_dict[field]

    model_config['llm_config'] = llm_config_dict
    # Remove llm_model_path as it's no longer needed
    if 'llm_model_path' in model_config:
        del model_config['llm_model_path']

    # Copy tokenizer files
    logger.info(f"Copying tokenizer files from: {cfg.model.llm_model_path}")
    tokenizer_source = Path(cfg.model.llm_model_path)
    if tokenizer_source.exists():
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        # Copy essential tokenizer files
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt',
            'special_tokens_map.json',
        ]
        for filename in tokenizer_files:
            src_file = tokenizer_source / filename
            if src_file.exists():
                shutil.copy2(src_file, tokenizer_dir / filename)
                logger.info(f"  Copied: {filename}")
        logger.info(f"Tokenizer files saved to: {tokenizer_dir}")
    else:
        logger.warning(f"Tokenizer source path not found: {tokenizer_source}")

    # Compose final inference config
    inference_config = {
        'model': model_config,
        'armel': armel_config,
    }

    # Extract model state dict
    logger.info("Extracting model parameters...")
    state_dict = checkpoint['state_dict']

    # Remove Lightning module prefix 'model.' if present
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    # Count parameters
    total_params = sum(p.numel() for p in model_state_dict.values())
    logger.info(f"Total parameters: {total_params:,}")

    # Save config YAML
    logger.info(f"Saving config to: {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(inference_config, f)

    # Save portable model weights
    save_dict = {
        'model_state_dict': model_state_dict,
        'training_info': {
            'global_step': global_step,
            'epoch': epoch,
        }
    }
    torch.save(save_dict, model_path)
    logger.info(f"Model weights saved to: {model_path}")

    logger.info("=" * 80)
    logger.info("Export complete! (Self-contained checkpoint)")
    logger.info(f"  - Model weights: {model_path}")
    logger.info(f"  - Config file: {config_path}")
    logger.info(f"  - Tokenizer: {tokenizer_dir}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Training info: epoch={epoch}, step={global_step}")
    logger.info("")
    logger.info("This checkpoint is self-contained and does NOT require llm_model_path!")
    logger.info("")
    logger.info("RFMel Configuration:")
    for key, value in model_config['rfmel'].items():
        logger.info(f"  - {key}: {value}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Export ARMel model weights and config from Lightning checkpoint")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to .ckpt file or directory with checkpoints')
    parser.add_argument('--output_path', type=str, required=True, help='Output .ckpt file or directory to save export')
    args = parser.parse_args()

    try:
        export_mel_checkpoint(args.ckpt_path, args.output_path)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
