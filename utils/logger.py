"""
Logging utilities for distributed training.

This module provides a centralized logging setup that automatically handles
distributed training scenarios (only rank 0 logs by default).
"""
import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger configured for distributed training.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger that only outputs from rank 0 in distributed settings
    """
    logger = logging.getLogger(name or __name__)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level based on rank
        rank = os.environ.get("RANK")
        local_rank = os.environ.get("LOCAL_RANK")
        
        # In distributed training, only rank 0 shows INFO and above
        # Other ranks only show WARNING and above
        if (rank is not None and int(rank) > 0) or (local_rank is not None and int(local_rank) > 0):
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(log_level)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def is_rank_zero() -> bool:
    """
    Check if current process is rank 0 (main process).
    
    Returns:
        True if rank 0 or not in distributed setting, False otherwise
    """
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    
    if rank is not None:
        return int(rank) == 0
    if local_rank is not None:
        return int(local_rank) == 0
    
    # Not in distributed setting
    return True
