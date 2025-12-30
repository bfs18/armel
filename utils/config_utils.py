"""
Utility functions for RFMel configuration management.
"""
from dataclasses import fields as dataclass_fields
from rfwave.mel_model import RFMelConfig


def get_rfmel_config_defaults():
    """
    Get default values from RFMelConfig dataclass definition.
    
    This ensures all default values come from a single source of truth,
    avoiding hardcoded defaults scattered across the codebase.
    
    Returns:
        dict: Dictionary mapping field names to their default values.
              Enum types are converted to their string values.
    """
    defaults = {}
    for field in dataclass_fields(RFMelConfig):
        default_value = field.default
        if hasattr(default_value, 'value'):
            default_value = default_value.value
        defaults[field.name] = default_value
    return defaults
