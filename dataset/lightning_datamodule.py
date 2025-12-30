"""
Lightning DataModule for ARMel training.
"""
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from dataset.arwave_dataset import ARWaveDataset
from dataset.data_collate import SampleCollator
from rfwave.mel_processor import MelConfig
from ar.special_tokens import audio_out_token, pad_token


class ARWaveDataModule(LightningDataModule):
    """Lightning DataModule for ARMel."""
    
    def __init__(self, cfg: DictConfig, tokenizer, head_config: MelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.head_config = head_config
        
        # Get token IDs
        audio_out_token_id, pad_token_id = tokenizer.convert_tokens_to_ids([audio_out_token, pad_token])
        
        # Create collator
        self.collator = SampleCollator(
            audio_out_token_id=audio_out_token_id,
            pad_token_id=pad_token_id,
            head_config=head_config,
            patch_size=cfg.model.patch_size,
            round_to=cfg.training.gradient_accumulation_steps * cfg.arwave.round_to,
            ignore_index=cfg.arwave.ignore_index
        )
    
    def setup(self, stage=None):
        """Set up datasets."""
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = ARWaveDataset(
                dataset_path=self.cfg.dataset.train_dataset_path,
                tokenizer=self.tokenizer,
                sample_rate=self.cfg.dataset.sample_rate,
                head_config=self.head_config,
                patch_size=self.cfg.model.patch_size,
                max_tokens=self.cfg.dataset.max_tokens
            )
            
            # Validation dataset (if provided)
            valid_dataset_path = self.cfg.dataset.get('valid_dataset_path', None)
            if valid_dataset_path:
                self.val_dataset = ARWaveDataset(
                    dataset_path=valid_dataset_path,
                    tokenizer=self.tokenizer,
                    sample_rate=self.cfg.dataset.sample_rate,
                    head_config=self.head_config,
                    patch_size=self.cfg.model.patch_size,
                    max_tokens=self.cfg.dataset.max_tokens
                )
            else:
                self.val_dataset = None
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.training.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.training.num_workers > 0 else False
        )
