#!/usr/bin/env python3
"""
ARMel (Mel-spectrogram) Training Script with Lightning.

Usage:
    python scripts/mel_train.py
    python scripts/mel_train.py experiment=debug
    python scripts/mel_train.py training.batch_size=8
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ar.two_opt_mel_lightning_module import ARMelTwoOptLightningModule
from dataset.lightning_datamodule import ARWaveDataModule
from utils.logger import get_logger
from utils.checkpoints import find_latest_checkpoint
from utils.callbacks import GradNormCallback

torch.set_float32_matmul_precision('high')
logger = get_logger(__name__)


def extract_wandb_run_id_from_directory(log_dir: Path):
    """Extract wandb run_id from the wandb directory if present."""
    wandb_dir = log_dir / "wandb"
    if not wandb_dir.exists():
        return None
    try:
        run_dirs = list(wandb_dir.glob("run-*"))
        if not run_dirs:
            return None
        latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        dir_name = latest_run_dir.name
        parts = dir_name.split('-')
        if len(parts) >= 3:
            run_id = '-'.join(parts[2:])
            logger.info(f"Found wandb run_id from directory: {run_id}")
            return run_id
    except Exception as e:
        logger.warning(f"Could not extract wandb run_id from directory: {e}")
    return None


def create_callbacks(log_dir: Path, save_steps: int, save_top_k: int = 5, save_last_k_steps: int = 3):
    callbacks = [
        ModelCheckpoint(
            dirpath=log_dir / "checkpoints",
            filename='armel-step_{step:07d}-val_{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=log_dir / "checkpoints",
            filename='armel-step_{step:07d}',
            monitor='step',
            mode='max',
            save_top_k=save_last_k_steps,
            every_n_train_steps=save_steps,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval='step'),
        GradNormCallback(),
    ]
    return callbacks


def create_profiler(log_dir: Path, enable_profiling: bool = False):
    if not enable_profiling:
        return None
    from lightning.pytorch.profilers import PyTorchProfiler
    rank = int(os.environ.get('LOCAL_RANK', 0))
    profiler_dir = log_dir / "profiler" / f"rank_{rank}"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Rank {rank}] Profiling enabled, output dir: {profiler_dir}")
    profiler = PyTorchProfiler(
        dirpath=str(profiler_dir),
        filename=f"profile_rank_{rank}",
        schedule=torch.profiler.schedule(
            wait=5,
            warmup=2,
            active=5,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        profile_memory_on_validation=False,
    )
    return profiler


@hydra.main(version_base=None, config_path="../configs", config_name="mel_train_config")
def main(cfg: DictConfig):
    logger.info("=" * 80)
    logger.info("Mel Training Configuration:")
    logger.info("=" * 80)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    seed_everything(cfg.training.seed, workers=True)

    # Create Lightning module (mel two-optimizer)
    logger.info("Initializing ARMel model...")
    lightning_module = ARMelTwoOptLightningModule(cfg)

    # Data module reuses the same implementation; pass mel head_config
    logger.info("Loading dataset...")
    datamodule = ARWaveDataModule(
        cfg=cfg,
        tokenizer=lightning_module.tokenizer,
        head_config=lightning_module.head_config
    )

    # Setup log directory
    log_dir = Path(cfg.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration once
    config_save_path = log_dir / "mel_train_config.yaml"
    if not config_save_path.exists():
        logger.info(f"Saving training configuration to: {config_save_path}")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(cfg, f)
    else:
        logger.info(f"Configuration file already exists: {config_save_path}")

    # WandB naming from folder
    wandb_run_name = cfg.training.get('wandb_run_name', None)
    if wandb_run_name is None:
        wandb_run_name = log_dir.name

    # Determine checkpoint path
    ckpt_path = cfg.training.get('ckpt_path', None)
    if ckpt_path:
        logger.info(f"Using specified checkpoint: {ckpt_path}")
    else:
        checkpoint_dir = log_dir / "checkpoints"
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    # Callbacks and profiler
    save_top_k = cfg.training.get('save_top_k', 5)
    save_last_k_steps = cfg.training.get('save_last_k_steps', 3)
    callbacks = create_callbacks(log_dir, cfg.training.save_steps, save_top_k, save_last_k_steps)
    profiler = create_profiler(log_dir, enable_profiling=cfg.training.get('enable_profiling', False))

    # WandB logger
    wandb_project = cfg.training.get('wandb_project', 'armel-training')
    wandb_run_id = None
    if ckpt_path:
        wandb_run_id = extract_wandb_run_id_from_directory(log_dir)
    wandb_logger_kwargs = {
        'project': wandb_project,
        'name': wandb_run_name,
        'save_dir': str(log_dir),
        'log_model': False,
    }
    if wandb_run_id:
        wandb_logger_kwargs['id'] = wandb_run_id
        wandb_logger_kwargs['resume'] = 'allow'
        logger.info(f"Resuming wandb run: {wandb_run_id}")
    wandb_logger = WandbLogger(**wandb_logger_kwargs)

    # Trainer
    trainer = Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='auto',
        devices='auto',
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        log_every_n_steps=cfg.training.logging_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler=profiler,
        deterministic=True,
        enable_checkpointing=True,
        default_root_dir=str(log_dir),
        num_sanity_val_steps=0 if profiler is not None else 10,
        val_check_interval=cfg.training.get("val_check_interval", None),
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", None),
        strategy='ddp_find_unused_parameters_true',
    )

    logger.info(f"Starting mel training for {cfg.training.max_steps} steps...")
    trainer.fit(lightning_module, datamodule, ckpt_path=ckpt_path)
    logger.info("Mel training completed!")


if __name__ == "__main__":
    main()


