"""
ARMel Configuration - Independent configuration for AR + Mel model.

This is a standalone config that does not depend on ARWaveConfig.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class ARMelConfig:
    """Configuration for ARMel model - core parameters for AR + Mel pipeline."""
    
    # Special token strings
    audio_out_token: str = "<|AUDIO_OUT|>"
    audio_out_bos_token: str = "<|audio_out_bos|>"
    audio_eos_token: str = "<|audio_eos|>"
    
    # Special token IDs (set at runtime from tokenizer)
    audio_out_token_id: Optional[int] = None
    audio_out_bos_token_id: Optional[int] = None
    audio_eos_token_id: Optional[int] = None
    
    # Patch and sequence settings
    patch_size: int = 8
    ignore_index: int = -100
    round_to: int = 8
    max_tokens: int = 2048
    
    # Audio settings
    sample_rate: int = 24000
    
    # Architecture settings
    use_skip_connection: bool = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ARMelConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> "ARMelConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

