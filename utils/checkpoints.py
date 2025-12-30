from pathlib import Path
from typing import Optional, Union

from utils.logger import get_logger

logger = get_logger(__name__)


def find_latest_checkpoint(checkpoint_dir: Union[Path, str]) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Priority:
    1) 'last.ckpt' if present
    2) Newest '*.ckpt' by modification time

    Returns:
        str path to checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Prefer 'last.ckpt' if present
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        logger.info(f"Found checkpoint: {last_ckpt}")
        return str(last_ckpt)

    # Otherwise pick newest *.ckpt by modification time
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None

    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found checkpoint: {latest_ckpt}")
    return str(latest_ckpt)
